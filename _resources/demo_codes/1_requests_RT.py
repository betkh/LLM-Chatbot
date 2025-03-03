import requests
import json

# Set up the base URL for the local Ollama API
url = "http://localhost:11434/api/chat"

# Define the model name (change to your preferred model)
model_name = "deepseek-r1"
# Welcome message
print(f"🤖 Chatbot using Ollama ({model_name}) - Type 'exit' to quit\n")

# Interactive chat loop
while True:
    user_input = input("You: ")

    # Exit the chat loop
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye! 👋")
        break

    # Define the payload with user input
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_input}]
    }

    # Send the HTTP POST request with streaming enabled
    response = requests.post(url, json=payload, stream=True)

    # Check  response status
    if response.status_code == 200:
        print("AI:", end=" ", flush=True)  # Show immediate AI response
        for line in response.iter_lines(decode_unicode=True):
            if line:  # Ignore empty lines
                try:
                    # Parse each line as JSON
                    json_data = json.loads(line)
                    # Extract and print the assistant's response
                    if "message" in json_data and "content" in json_data["message"]:
                        print(json_data["message"]
                              ["content"], end="", flush=True)
                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")
        print()  # Ensure proper formatting after response
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
