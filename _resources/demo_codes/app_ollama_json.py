import re
import ollama
import pandas as pd
from flask import Response, jsonify, Flask, render_template, request

app = Flask(__name__)

# Define model name [mistral, deepseek-r1, llama2]
MODEL_NAME = "deepseek-r1"


@app.route("/")
def index():
    return render_template('chat.html')


def extract_csv_from_response(response_text):
    """
    Extracts CSV data enclosed in triple backticks from the response text.
    """
    match = re.search(r"```(?:csv)?\n(.*?)\n```", response_text, re.DOTALL)
    return match.group(1).strip() if match else None


def format_json_from_response(response_text):
    """
    Converts unstructured text data into a JSON format.
    Expected formats:
    - "California: 418362000, Texas: 275930000"
    - "California - 418362000 | Texas - 275930000"
    - CSV-formatted response
    """
    data = []

    # Try to extract structured key-value pairs
    matches = re.findall(r"(\w+)\s*[:,-]\s*([\d,]+)", response_text)

    for match in matches:
        category = match[0].strip()
        value = int(match[1].replace(",", "").strip())  # Convert to int
        data.append({"category": category, "value": value})

    return data if data else None  # Return None if no structured data is found


def get_Chat_response_only(input_text, model_name):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": input_text}
            ]
        )

        if 'message' not in response or 'content' not in response['message']:
            return jsonify({"error": "Invalid response from Ollama"}), 500

        response_text = response['message']['content']

        # Extract CSV or reformat response into JSON
        csv_data = extract_csv_from_response(response_text)
        json_data = format_json_from_response(
            csv_data if csv_data else response_text)

        if json_data:
            return jsonify(json_data)

        # Fallback: Stream normal response as plain text
        def generate():
            new_list = []
            in_think_block = False

            for token in response_text.split():  # Simulate tokenized streaming
                if "<think>" in token:
                    in_think_block = True
                if "</think>" in token:
                    in_think_block = False
                    token = token.split("</think>", 1)[-1]

                if not in_think_block and token:
                    print(token)  # Debugging
                    new_list.append(token)
                    yield token + " "  # Stream only valid output
                    print(token)

            print(new_list)

        return Response(generate(), content_type='text/plain')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get", methods=["GET", "POST"])
def chat():
    input_text = request.form["msg"]
    print("Input message:", input_text, "\n\n")
    return get_Chat_response_only(input_text, MODEL_NAME)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
