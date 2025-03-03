"""
demo: interact using ollama 
"""
import ollama

response = ollama.chat(
    model='llama2',  # Change this to 'deepseek-r1:7b' if needed
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "try again and avoid text just guve me the data in csv format"}
    ]
)

print(response['message']['content'])
