import json
import re
import requests
from flask import Response, jsonify
from flask import Flask, render_template, request
# from modules.genericResponse import get_Chat_response_generic


def get_Chat_response_only(input, MODEL_NAME, OLLAMA_API_URL):
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": input}]
        }

        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch response from Ollama", "status": response.status_code}), response.status_code

        def generate():

            new_list = []
            full_response = []
            for line in response.iter_lines(decode_unicode=True):
                if line:

                    try:
                        json_data = json.loads(line)
                        token = json_data.get("message", {}).get(
                            "content", "")

                        # isolate the thik block to ignore reasoning essage
                        # Ignore everything inside <think>...</think>
                        if "<think>" in token:
                            in_think_block = True
                        if "</think>" in token:
                            in_think_block = False
                            # Get text after </think>
                            token = token.split("</think>", 1)[-1].strip()

                        if not in_think_block and token:
                            print(token)  # Debugging
                            new_list.append(token.strip())
                            yield token  # Stream only valid output

                        # if "message" in json_data and "content" in json_data["message"]:

                        #     token = json_data["message"]["content"]
                        #     print(token)

                            # yield token
                    except json.JSONDecodeError:
                        yield f"\nFailed to parse line: {line}"
            print(new_list)

        return Response(generate(), content_type='text/plain')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def extract_json_from_response(response):
    """
    Extracts and parses JSON data from a given response string.

    Parameters:
        response (str): The string containing embedded JSON.

    Returns:
        list or dict: Parsed JSON data if successful, otherwise None.
    """
    # Extract JSON content using regex
    match = re.search(r"```json(.*?)```", response, re.DOTALL)

    if match:
        json_text = match.group(1).strip()  # Extract JSON part
        try:
            # Parse and return JSON
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None
    else:
        print("No JSON found in response.")
        return None


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

            for line in response.iter_lines(decode_unicode=True):
                if line:

                    try:
                        json_data = json.loads(line)
                        if "message" in json_data and "content" in json_data["message"]:

                            token = json_data["message"]["content"]
                            # print(token)
                            yield token
                    except json.JSONDecodeError:
                        yield f"\nFailed to parse line: {line}"

        result = Response(generate(), content_type='text/plain')

        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Flask Application
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
