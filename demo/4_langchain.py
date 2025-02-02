"""
using langchain 

"""

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

# Define the model
chat_model = ChatOllama(model="deepseek-r1")

# Send messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(
        content="Write a python code that adds two numbers and returns the sum")
]

# Get response
response = chat_model(messages)
print(response.content)
