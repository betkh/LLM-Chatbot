"""
Interactive mode

"""


from langchain_community.chat_models import ChatOllama

# Enable streaming
chat_model = ChatOllama(model="deepseek-r1", streaming=True)

# Create a conversation loop


def chat_loop():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = chat_model.invoke(user_input)
        print("AI:", response.content)


# Run the chat loop
chat_loop()
