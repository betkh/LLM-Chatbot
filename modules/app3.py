import json
import os
import requests
from flask import Response, jsonify
from sacremoses import MosesDetokenizer
from flask import Flask, render_template, request
from modules.genericResponse import get_Chat_response_generic

# from modules.formattedResponse import get_Chat_response_formatted
from modules.responseOnly import get_Chat_response_only


app = Flask(__name__)


# Set up the base URL for the local Ollama API
OLLAMA_API = "http://localhost:11434/api/chat"

# define Model NAME    [mistral, deepseek-r1, llama2 ]
# MODEL_NAME = "llama2"
# MODEL_NAME = "mistral"
MODEL_NAME = "deepseek-r1"


@app.route("/")
def index():
    return render_template('chat.html')


def get_Chat_response_generic(input, modelName, ollama_api):
    try:
        payload = {
            "model": modelName,
            "messages": [{"role": "user", "content": input}]
        }

        response = requests.post(ollama_api, json=payload, stream=True)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch response from Ollama", "status": response.status_code}), response.status_code

        def generate():
            new_list = []
            response_text = ""  # Store response here
            for line in response.iter_lines(decode_unicode=True):
                if line:

                    try:
                        json_data = json.loads(line)
                        if "message" in json_data and "content" in json_data["message"]:

                            token = json_data["message"]["content"]
                            # print(token)
                            new_list.append(token.strip())
                            response_text += token + " "  # Append to string
                            yield token
                    except json.JSONDecodeError:
                        yield f"\nFailed to parse line: {line}"

            print(new_list)

            # detokinizer
            # Initialize detokenizer
            detokenizer = MosesDetokenizer()

            # Detokenize list
            sentence = detokenizer.detokenize(new_list, return_str=True)

            print(sentence)

            # Save to a text file
            # Define the file path in the current directory
            # file_path = os.path.join(os.getcwd(), "chat_response.txt")
            # with open(file_path, "w", encoding="utf-8") as file:
            #     file.write(response_text.strip())

        result = Response(generate(), content_type='text/plain')

        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get", methods=["GET", "POST"])
def chat():
    input = request.form["msg"]

    # raw input message
    print("Input message: ")

    # prompt engineering to steer the output e.g. tabular format, csv

    print(input, "\n\n")
    return get_Chat_response_only(input, MODEL_NAME, OLLAMA_API)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
