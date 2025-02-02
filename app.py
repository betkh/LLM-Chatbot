from flask import Flask, render_template, request
from modules.genericResponse import get_Chat_response_generic
from modules.formattedResponse import get_Chat_response_formatted
from modules.responseOnly import get_Chat_response_only


app = Flask(__name__)


# Set up the base URL for the local Ollama API
OLLAMA_API = "http://localhost:11434/api/chat"

# define Model NAME    [mistral, deepseek-r1, llama2 ]
MODEL_NAME = "llama2"
# MODEL_NAME = "mistral"
# MODEL_NAME = "deepseek-r1"


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    print("Input message: ")

    print(input, "\n\n")
    return get_Chat_response_generic(input, MODEL_NAME, OLLAMA_API)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
