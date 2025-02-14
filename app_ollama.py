import ollama
from flask import Response, jsonify
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__)

# Define model name [mistral, deepseek-r1, llama2]
MODEL_NAME = "deepseek-r1"


@app.route("/")
def index():
    return render_template('chat.html')


def get_Chat_response_generic(input_text, model_name):
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
        return Response(response_text, content_type='text/plain')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
