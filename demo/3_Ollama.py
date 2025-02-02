"""
demo: interact using ollama 
"""
import ollama

response = ollama.chat(
    model='deepseek-r1',  # Change this to 'deepseek-r1:7b' if needed
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a python code that adds two numbers and returns the sum"}
    ]
)

print(response['message']['content'])
