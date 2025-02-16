from flask import Response, send_file
import re
import json
import requests
from flask import Response, jsonify
from flask import Flask, render_template, request

# from modules.genericResponse import get_Chat_response_generic


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


def get_Chat_response_only(input, MODEL_NAME, OLLAMA_API_URL):
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": input}]
        }

        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code != 200:
            return Response(json.dumps({"error": "Failed to fetch response from Ollama", "status": response.status_code}),
                            status=response.status_code,
                            content_type='application/json')

        # Collect the full response
        full_response = []
        in_think_block = False  # Track if inside a <think> block

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    token = json_data.get("message", {}).get("content", "")
                    print(token)

                    # Ignore everything inside <think>...</think>
                    if "<think>" in token:
                        in_think_block = True
                    if "</think>" in token:
                        in_think_block = False
                        # Get text after </think>
                        token = token.split("</think>", 1)[-1].strip()

                    if not in_think_block and token:
                        full_response.append(token.strip())

                except json.JSONDecodeError:
                    return Response(json.dumps({"error": f"Failed to parse line: {line}"}),
                                    content_type='application/json', status=500)

        # Convert full response to a single string
        final_response = "".join(full_response)
        print("Final Response:", final_response)

        # Extract JSON from the chatbot's response text
        data = extract_json_from_response(final_response)
        print("Extracted JSON Data:", data)

        # Return as a proper JSON response
        return Response(jsonify(data), content_type='application/text')

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, content_type='application/json')


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
