import json
import requests
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, Response, jsonify, render_template, request

CHAT_HISTORY_FILE = "chat_history.json"
EXTRACTED_JSON_FILE = "extracted_data.json"
PROMPT_PATTERNS_FILE = "prompt_patterns.json"
PROMPT_TEMPLATE_FILE = "prompt_steering_template.json"

CONTEXT_LIMIT = 5  # Number of previous interactions to include in context

app = Flask(__name__)

# Set up the base URL for the local Ollama API
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL_NAME = "llama2"


def load_chat_history():
    """Loads chat history from the JSON file."""
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Return empty list if no history exists


def save_chat_history(history):
    """Saves updated chat history to the JSON file."""
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(history, file, indent=4)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def get_chat_context(input_text):
    """
    Retrieves the last few chat interactions as context for the bot.
    Integrates the best prompt template into the system message.
    """
    history = load_chat_history()
    context = history[-CONTEXT_LIMIT:]  # Keep only recent interactions

    # try:
    #     with open(PROMPT_TEMPLATE_FILE, "r", encoding="utf-8") as file:
    #         prompt_data = json.load(file)
    # except (FileNotFoundError, json.JSONDecodeError):
    #     print(f"Error loading {PROMPT_TEMPLATE_FILE}. Using default template.")
    #     prompt_data = {}

    # if prompt_data:
    #     template_names = list(prompt_data.keys())
    #     template_prompts = [data["prompt_template"]
    #                         for data in prompt_data.values()]

    #     print("TEMPLATE PROMPT ->", template_prompts)

    #     model = SentenceTransformer("all-MiniLM-L6-v2")
    #     input_embedding = model.encode([input_text])
    #     template_embeddings = model.encode(template_prompts)
    #     similarities = cosine_similarity(
    #         input_embedding, template_embeddings)[0]

    #     print("SIMILARITIES  ->", similarities)

    #     best_prompt_key = template_names[np.argmax(similarities)]
    #     print("best prompt key is: ", best_prompt_key)
    #     best_prompt = prompt_data[best_prompt_key]["prompt_template"]

    #     print("THE BEST PROMPT ->", best_prompt)

    # else:
    #     best_prompt = "You are a helpful AI assistant."

    system_message = {
        "role": "You are a structured data extraction assistant. Your task is to provide responses strictly in CSV format, ensuring that outputs are free from additional text, explanations, markdown formatting, or extra symbols.",
        "content": f"Use prior context to answer questions correctly. The output MUST STRICTLY follow comma separtaed values! and Do not include additional text such as 'here is your response ...'"
    }

    formatted_context = [system_message]
    for entry in context:
        formatted_context.append(
            {"role": "user", "content": f"User: {entry['User']}"})
        formatted_context.append(
            {"role": "assistant", "content": f"Bot: {entry['Bot']}"})

    return formatted_context


def extract_json_from_bot_response(bot_response):
    """
    Extracts JSON data from a bot response.
    """
    match = re.search(r"```json(.*?)```", bot_response, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
        try:
            extracted_json = json.loads(json_text)
            return extracted_json if isinstance(extracted_json, list) else None
        except json.JSONDecodeError:
            print("Error parsing JSON from bot response.")
    return None


def save_extracted_data(bot_response):
    """
    Extracts and saves structured data from the bot response into a JSON file.
    """
    extracted_data = extract_json_from_bot_response(bot_response)
    if extracted_data:
        try:
            with open(EXTRACTED_JSON_FILE, "w") as json_file:
                json.dump(extracted_data, json_file, indent=4)
            print(f"Extracted data saved to {EXTRACTED_JSON_FILE}")
        except Exception as e:
            print(f"Error saving extracted data: {e}")
    return extracted_data


def get_chat_response(input_text):
    """
    Sends user input to Ollama API, including chat history for context.
    """
    try:
        context = get_chat_context(input_text)
        context.append({"role": "user", "content": input_text})

        payload = {
            "model": MODEL_NAME,
            "messages": context
        }

        response = requests.post(OLLAMA_API, json=payload, stream=True)

        if response.status_code != 200:
            return Response(
                json.dumps(
                    {"error": "Failed to fetch response from Ollama", "status": response.status_code}),
                status=response.status_code,
                content_type="application/json"
            )

        # Process response while handling <think> blocks
        full_response = []
        in_think_block = False

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    token = json_data.get("message", {}).get("content", "")

                    if "<think>" in token:
                        in_think_block = True
                    if "</think>" in token:
                        in_think_block = False
                        token = token.split("</think>", 1)[-1].strip()

                    if not in_think_block and token:
                        full_response.append(token)

                except json.JSONDecodeError:
                    return Response(json.dumps({"error": f"Failed to parse response: {line}"}), 500)

        final_response = "".join(full_response)
        print("Final Response:", final_response)

        history = load_chat_history()
        history.append({"User": input_text, "Bot": final_response})
        save_chat_history(history)

        save_extracted_data(final_response)

        return Response(final_response, content_type="text/plain")

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, content_type="application/json")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/clear")
def clear_conversation():
    """
    Clears the chat history JSON file.
    """
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump([], file)
        return jsonify({"status": "Conversation cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_chat_history():
    """
    Returns the saved chat history as JSON.
    """
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            chat_history = json.load(file)
        return jsonify(chat_history)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get", methods=["POST"])
def chat():
    """
    Handles user input and returns the chatbot's response.
    """
    input_text = request.form.get("msg", "").strip()

    if not input_text:
        return jsonify({"error": "No message received"}), 400

    return get_chat_response(input_text)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
