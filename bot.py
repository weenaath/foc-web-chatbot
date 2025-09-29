import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_bot(user_message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",   # you can switch to gpt-4.1 or gpt-4.1-turbo
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("🤖 Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        answer = chat_with_bot(user_input)
        print("Bot:", answer)
