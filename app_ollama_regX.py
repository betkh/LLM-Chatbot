import re
import ollama
import pandas as pd
from flask import Response, jsonify, Flask, render_template, request
from io import StringIO

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


def format_csv_from_response(response_text):
    """
    Converts unstructured text data into a two-column CSV format if needed.
    Expected format in text: "California: 418362000, Texas: 275930000"
    """
    data = []

    # Try to extract structured key-value pairs
    matches = re.findall(r"(\w+)\s*[:,-]\s*([\d,]+)", response_text)

    for match in matches:
        state = match[0].strip()
        emissions = match[1].replace(",", "").strip()
        data.append([state, emissions])

    # Convert to DataFrame if valid
    if data:
        df = pd.DataFrame(data, columns=["State", "CO2 Emissions"])
        output = StringIO()
        df.to_csv(output, index=False)
        print(df)
        return output.getvalue()

    return None  # No structured data found


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

        # Extract or format CSV data
        csv_data = extract_csv_from_response(
            response_text) or format_csv_from_response(response_text)
        if csv_data:
            return Response(csv_data, content_type='text/csv')

        # Fallback: Stream normal response
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
