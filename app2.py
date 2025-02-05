# from flask import Flask, render_template, request, jsonify, session
# import requests
# import json

# app = Flask(__name__)
# app.secret_key = "supersecretkey"  # Required for session management

# # Set up the base URL for the local Ollama API
# OLLAMA_API = "http://localhost:11434/api/chat"

# # Define Model NAME [mistral, deepseek-r1, llama2]
# MODEL_NAME = "llama2"


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get", methods=["POST"])
# def chat():
#     msg = request.form["msg"]

#     # Ensure session chat history is initialized
#     if "chat_history" not in session:
#         session["chat_history"] = [
#             {"role": "system", "content": "You are a helpful AI assistant."}]

#     # Append user message to history
#     session["chat_history"].append({"role": "user", "content": msg})

#     # Generate AI response
#     ai_response = get_chat_response(
#         session["chat_history"], MODEL_NAME, OLLAMA_API)

#     # Append AI response to history
#     session["chat_history"].append(
#         {"role": "assistant", "content": ai_response})
#     session.modified = True  # Ensure session is updated

#     return jsonify({"response": ai_response})  # Return JSON response


# def get_chat_response(chat_history, model_name, ollama_api):
#     try:
#         payload = {
#             "model": model_name,
#             "messages": chat_history  # Send full conversation history
#         }

#         response = requests.post(ollama_api, json=payload)

#         if response.status_code != 200:
#             return f"Error: Failed to fetch response from Ollama (Status: {response.status_code})"

#         json_data = response.json()
#         if "message" in json_data and "content" in json_data["message"]:
#             return json_data["message"]["content"]

#         return "Error: No valid response received."

#     except Exception as e:
#         return f"Error: {str(e)}"


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
