import json
import requests
import re
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, Response, send_file, jsonify, render_template, request

CHAT_HISTORY_FILE = "chat_history.json"


def load_chat_history():
    """ Loads chat history from the JSON file. """
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Return empty list if no history exists


def save_chat_history(history):
    """ Saves updated chat history to the JSON file. """
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(history, file, indent=4)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def get_chat_context():
    """
    Retrieves the last few chat interactions as context for the bot.

    Returns:
        list: List of messages formatted for Ollama API input.
    """
    history = load_chat_history()

    # Limit context to the most recent `CONTEXT_LIMIT` exchanges
    context = history[-CONTEXT_LIMIT:]

    formatted_context = [
        {"role": "system", "content": "You are a helpful AI assistant. Use prior context to answer follow-up questions correctly."}]

    for entry in context:
        formatted_context.append(
            {"role": "user", "content": f"User: {entry['User']}"})
        formatted_context.append(
            {"role": "assistant", "content": f"Bot: {entry['Bot']}"})

    print(formatted_context)

    return formatted_context


def extract_json_from_bot_response(bot_response):
    """
    Extracts JSON data from a bot response.

    Parameters:
        bot_response (str): The response text containing embedded JSON.

    Returns:
        list: Extracted structured data as a list of dictionaries, or None if not found.
    """
    # Use regex to extract JSON content inside triple backticks
    match = re.search(r"```json(.*?)```", bot_response, re.DOTALL)
    if match:
        json_text = match.group(1).strip()

        try:
            # Convert extracted text into valid JSON format
            extracted_json = json.loads(json_text)

            if isinstance(extracted_json, list):
                return extracted_json  # Ensure it's a list of dicts

        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON due to error: {e}")

    return None


def save_extracted_data(bot_response):
    """
    Extracts and saves structured data from the bot response into a JSON file.

    Parameters:
        bot_response (str): The bot's response containing structured JSON data.

    Returns:
        list: Extracted JSON data if successful, None otherwise.
    """
    extracted_data = extract_json_from_bot_response(bot_response)

    if extracted_data:
        try:
            # Save extracted data to a JSON file
            with open(EXTRACTED_JSON_FILE, "w") as json_file:
                json.dump(extracted_data, json_file, indent=4)
            print(f"Extracted data saved to {EXTRACTED_JSON_FILE}")
        except Exception as e:
            print(f"Error saving extracted data: {e}")

    return extracted_data


def load_prompt_patterns():
    """Loads prompt patterns from the JSON file."""
    try:
        with open(PROMPT_PATTERNS_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading prompt patterns: {e}")
        return {}


def get_Chat_response_only(input_text, MODEL_NAME, OLLAMA_API_URL):
    """
    Sends user input to Ollama API, including chat history for context.

    Returns:
        Response: Flask Response containing the bot's reply.
    """
    try:
        context = get_chat_context()  # Retrieve conversation history
        print("This is chat history below: \n", context)

        print("\n This is chat history above: \n")

        # Append new user input to the context
        context.append({"role": "user", "content": input_text})

        payload = {
            "model": MODEL_NAME,
            "messages": context
        }

        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code != 200:
            return Response(json.dumps({"error": "Failed to fetch response from Ollama", "status": response.status_code}),
                            status=response.status_code,
                            content_type='application/json')

        # Collect response
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
                    return Response(json.dumps({"error": f"Failed to parse line: {line}"}), 500)

        # Convert full response to a single string
        final_response = "".join(full_response)
        print("Final Response:", final_response)

        # Update chat history
        history = load_chat_history()
        history.append({"User": input_text, "Bot": final_response})
        save_chat_history(history)

        # Extract structured data from bot response
        save_extracted_data(final_response)

        return Response(final_response, content_type='text/plain')

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, content_type='application/json')


app = Flask(__name__)

# Set up the base URL for the local Ollama API
OLLAMA_API = "http://localhost:11434/api/chat"
MODEL_NAME = "llama2"
# MODEL_NAME = "mistral"
# MODEL_NAME = "deepseek-r1"
# MODEL_NAME = "deepseek-r1:32b"

# Define the output file for extracted JSON data
EXTRACTED_JSON_FILE = "extracted_data.json"
CONTEXT_LIMIT = 5  # Number of previous interactions to include in context


# Load the prompt patterns from the JSON file
PROMPT_PATTERNS_FILE = "prompt_patterns.json"


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/clear")
def clear_conversation():
    """
    Clears the chat history JSON file.
    """
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump([], file)  # Empty list to clear history
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
        # Return an empty list if history is missing or corrupted
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get", methods=["POST"])
def chat():
    """
    Handles user input and returns the chatbot's response.
    """
    input_text = request.form["msg"]

    return get_Chat_response_only(input_text, MODEL_NAME, OLLAMA_API)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
