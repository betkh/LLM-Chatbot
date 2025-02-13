from flask import Flask, render_template, request
from modules.genericResponse import get_Chat_response_generic
# from modules.formattedResponse import get_Chat_response_formatted
# from modules.responseOnly import get_Chat_response_only


from flask import Flask, render_template, request, jsonify
from modules.genericResponse import get_Chat_response_generic

app = Flask(__name__)

# Set up the base URL for the local Ollama API
OLLAMA_API = "http://localhost:11434/api/chat"

# Define Model NAME [mistral, deepseek-r1, llama2]
MODEL_NAME = "llama2"


def enhance_prompt(user_input, model_name):
    """
    Enhances the user input prompt to guide the model towards a professional, quantitative, and CSV formatted response.
    """
    enhanced_prompt = (
        f"You are an expert in this domain. Please provide a detailed, quantitative response to the following query. "
        f"Ensure the output is in CSV format with clear, comma-separated values.\n\n"
        f"Query: {user_input}\n"
        f"Generate multiple variations of the response and combine them for a comprehensive result."
    )
    return enhanced_prompt


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_input = request.form["msg"]

    # Display raw input message for debugging
    print("Input message:", user_input, "\n\n")

    # Enhance the prompt
    enhanced_input = enhance_prompt(user_input, MODEL_NAME)

    # Get response from the model
    response = get_Chat_response_generic(
        enhanced_input, MODEL_NAME, OLLAMA_API)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
