"""
Demo: Interact using Ollama with memory of previous chats
"""

import ollama

# Store chat history
chat_history = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]


def chat_with_ollama(user_input):
    global chat_history

    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Get response from Ollama
    response = ollama.chat(
        model='llama2',  # Change this to 'deepseek-r1:7b' if needed
        messages=chat_history
    )

    # Append assistant response to chat history
    assistant_message = response['message']['content']
    chat_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message


# Main loop for user interaction
while True:
    user_input = input("You: ")  # Accept user input first
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    response_text = chat_with_ollama(user_input)
    print("AI:", response_text)
