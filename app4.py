# import json
# import requests
# import re
# import matplotlib.pyplot as plt
# import pandas as pd
# from flask import Flask, Response, send_file, jsonify, render_template, request
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer


# def load_chat_history():
#     """ Loads chat history from the JSON file.
#         extracts chat history to provide as a context for a followup conv.
#     """
#     try:
#         with open(CHAT_HISTORY_FILE, "r") as file:
#             return json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError):
#         return []  # Return empty list if no history exists


# def save_chat_history(history):
#     """ Saves updated chat history to the JSON file.
#         Chat history is reset every time page is reloaded.
#     """
#     try:
#         with open(CHAT_HISTORY_FILE, "w") as file:
#             json.dump(history, file, indent=4)
#     except Exception as e:
#         print(f"Error saving chat history: {e}")


# def get_chat_context():
#     """
#     enables to remeber follow up conversation
#     Retrieves the last few chat interactions as context for the bot.

#     Returns:
#         list: List of messages formatted for Ollama API input.
#     """
#     history = load_chat_history()

#     # Limit context to the most recent `CONTEXT_LIMIT` exchanges
#     context = history[-CONTEXT_LIMIT:]

#     formatted_context = [
#         {"role": "system", "content": "You are a helpful AI assistant. Use prior context to answer follow-up questions correctly."}]

#     for entry in context:
#         formatted_context.append(
#             {"role": "user", "content": f"User: {entry['User']}"})
#         formatted_context.append(
#             {"role": "assistant", "content": f"Bot: {entry['Bot']}"})

#     return formatted_context


# # ================================================================================================
# # =================================== QUANTITATIVE DATA EXTRACTION & SAVING ======================
# # ================================================================================================


# def extract_json_from_bot_response(bot_response):
#     """
#     Extracts JSON data from a bot response.
#     Useful to extract quantitative data for analysis / Visualization purpose.

#     Parameters:
#         bot_response (str): The response text containing embedded JSON.

#     Returns:
#         list: Extracted structured data as a list of dictionaries, or None if not found.
#     """
#     # Use regex to extract JSON content inside triple backticks
#     match = re.search(r"```json(.*?)```", bot_response, re.DOTALL)
#     if match:
#         json_text = match.group(1).strip()

#         try:
#             # Convert extracted text into valid JSON format
#             extracted_json = json.loads(json_text)

#             if isinstance(extracted_json, list):
#                 return extracted_json  # Ensure it's a list of dicts

#         except json.JSONDecodeError as e:
#             print(f"Skipping invalid JSON due to error: {e}")

#     return None


# def save_extracted_data(bot_response):
#     """
#     Extracts and saves structured data from the bot response into a JSON file.

#     Saves the extracted quantitative data into JSON file to be used for analysis purpose.

#     Parameters:
#         bot_response (str): The bot's response containing structured JSON data.

#     Returns:
#         list: Extracted JSON data if successful, None otherwise.
#     """
#     extracted_data = extract_json_from_bot_response(bot_response)

#     if extracted_data:
#         try:
#             # Save extracted data to a JSON file
#             with open(EXTRACTED_JSON_FILE, "w") as json_file:
#                 json.dump(extracted_data, json_file, indent=4)
#             print(f"Extracted data saved to {EXTRACTED_JSON_FILE}")
#         except Exception as e:
#             print(f"Error saving extracted data: {e}")

#     return extracted_data


# # ================================================================================================
# # =================================== PROMPT ENGINEERING SECTION  ================================
# # ================================================================================================

# def load_prompt_patterns():
#     """Loads prompt patterns from the JSON file."""
#     try:
#         with open(PROMPT_PATTERNS_FILE, "r") as file:
#             return json.load(file)
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         print(f"Error loading prompt patterns: {e}")
#         return {}


# # read the steering prompt pattern
# def load_prompt_data(file_path):
#     """
#     Reads the prompt data from a prompt_steering_template.json and
#     extracts template names, prompts, and formats.


#     """
#     with open(file_path, "r", encoding="utf-8") as file:
#         prompt_data = json.load(file)

#     # Extract template names, prompts, and formats
#     template_names = list(prompt_data.keys())
#     template_prompts = [data["prompt_template"]
#                         for data in prompt_data.values()]
#     prompt_formats = [data["output_format"] for data in prompt_data.values()]

#     return template_names, template_prompts, prompt_formats


# def extract_best_prompt(input_text, template_prompts, template_names):
#     """
#     Extracts the best matching prompt template based on similarity calculation.
#     Returns the key of the best matching prompt.

#     # extract the best prompt after similarity calculation
#     """
#     # Calculate similarity scores
#     similarities = calculate_similarity(input_text, template_prompts)

#     # Find the index of the best matching prompt
#     best_match_index = np.argmax(similarities)

#     # Return the key of the best matching prompt
#     return template_names[best_match_index]


# def subset_best_prompt(file_path, best_prompt_key):
#     """
#     Extracts the best prompt and its format using the identified prompt key.

#     subset the most fitting prompt pattern from the list
#     """
#     with open(file_path, "r", encoding="utf-8") as file:
#         prompt_data = json.load(file)

#     # Extract the best prompt and format
#     best_prompt = prompt_data[best_prompt_key]["prompt_template"]
#     best_format = prompt_data[best_prompt_key]["output_format"]

#     return best_prompt, best_format


# def create_prompt_template_json(best_prompt, best_format, output_file="best_prompt_template.json"):
#     """
#     Creates a new JSON file containing the best prompt template and output format.

#     create the best prompt based on subset output
#     """
#     prompt_template = {
#         "prompt_template": best_prompt,
#         "output_format": best_format
#     }

#     # Write to a new JSON file
#     with open(output_file, "w", encoding="utf-8") as file:
#         json.dump(prompt_template, file, indent=4)

#     print(f"Best prompt template saved to {output_file}")


# # calculate similarity between input prompt with engineered prompt
# def calculate_similarity(input_text, template_prompts):
#     """
#     Calculates similarity scores between the input text and prompt templates.
#     Returns a numpy array of similarity scores.

#     Helps to select the best prompt based on user input
#     """
#     # Encode input text and prompt templates

#     # Load sentence-transformers model
#     model = SentenceTransformer('all-MiniLM-L6-v2')

#     input_embedding = model.encode([input_text])
#     template_embeddings = model.encode(template_prompts)

#     # Compute similarity scores
#     return cosine_similarity(input_embedding, template_embeddings)[0]


# def get_Chat_response_only(input_text, MODEL_NAME, OLLAMA_API_URL):

#     try:
#         # Get the most relevant prompt pattern
#         relevant_prompt = get_most_relevant_prompt(input_text)

#         # Format the prompt using the selected pattern
#         formatted_prompt = relevant_prompt["prompt_template"].format(
#             location_A="Location A",  # Replace with actual values
#             location_B="Location B",  # Replace with actual values
#             occasion="Birthday",      # Replace with actual values
#             output_format=json.dumps(
#                 relevant_prompt["output_format"], indent=2)
#         )

#         # Send the formatted prompt to the Ollama API
#         payload = {
#             "model": MODEL_NAME,
#             "messages": [{"role": "user", "content": formatted_prompt}]
#         }

#         response = requests.post(OLLAMA_API_URL, json=payload)

#         if response.status_code != 200:
#             return Response(json.dumps({"error": "Failed to fetch response from Ollama", "status": response.status_code}),
#                             status=response.status_code,
#                             content_type='application/json')

#         # Process the response
#         full_response = []
#         for line in response.iter_lines(decode_unicode=True):
#             if line:
#                 try:
#                     json_data = json.loads(line)
#                     token = json_data.get("message", {}).get("content", "")
#                     full_response.append(token)
#                 except json.JSONDecodeError:
#                     return Response(json.dumps({"error": f"Failed to parse line: {line}"}), 500)

#         # Convert full response to a single string
#         final_response = "".join(full_response)
#         print("Final Response:", final_response)

#         return Response(final_response, content_type='text/plain')

#     except Exception as e:
#         return Response(json.dumps({"error": str(e)}), status=500, content_type='application/json')


# app = Flask(__name__)

# # Set up the base URL for the local Ollama API
# OLLAMA_API = "http://localhost:11434/api/chat"
# MODEL_NAME = "llama2"
# # MODEL_NAME = "mistral"
# # MODEL_NAME = "deepseek-r1"
# # MODEL_NAME = "deepseek-r1:32b"
# CHAT_HISTORY_FILE = "chat_history.json"
# # Define the output file for extracted JSON data
# EXTRACTED_JSON_FILE = "extracted_data.json"
# CONTEXT_LIMIT = 5  # Number of previous interactions to include in context


# # Load the prompt patterns from the JSON file
# PROMPT_PATTERNS_FILE = "prompt_steering_template.json"


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/clear")
# def clear_conversation():
#     """
#     Clears the chat history JSON file.
#     """
#     try:
#         with open(CHAT_HISTORY_FILE, "w") as file:
#             json.dump([], file)  # Empty list to clear history
#         return jsonify({"status": "Conversation cleared"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/history", methods=["GET"])
# def get_chat_history():
#     """
#     Returns the saved chat history as JSON.
#     """
#     try:
#         with open(CHAT_HISTORY_FILE, "r") as file:
#             chat_history = json.load(file)
#         return jsonify(chat_history)
#     except (FileNotFoundError, json.JSONDecodeError):
#         # Return an empty list if history is missing or corrupted
#         return jsonify([])
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route("/get", methods=["POST"])
# def chat():
#     """
#     Handles user input and returns the chatbot's response.
#     """
#     input_text = request.form["msg"]

#     return get_Chat_response_only(input_text, MODEL_NAME, OLLAMA_API)


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
