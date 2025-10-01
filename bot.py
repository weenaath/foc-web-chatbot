import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load faculty content
with open("faculty_content.txt", "r", encoding="utf-8") as f:
    FACULTY_CONTENT = f.read()

def chat_with_bot(user_input):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # Groq recommended fast model
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant that answers questions "
                    "about the Faculty of Computing at University of Sri Jayewardenepura. "
                    "Only use the following information as your knowledge base. Carefully read and understand before answering:\n\n"
                    + FACULTY_CONTENT
                )},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2  # lower = more focused, less random
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"
    
def polish_answer(answer: str) -> str:
    # Very light cleanup
    answer = answer.strip()
    # Capitalize first letter if missing
    if answer and not answer[0].isupper():
        answer = answer[0].upper() + answer[1:]
    # Ensure it ends with proper punctuation
    if answer and answer[-1] not in ".!?":
        answer += "."
    return answer


if __name__ == "__main__":
    print("🤖 Faculty Chatbot (Groq API) is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        answer = chat_with_bot(user_input)
        answer = polish_answer(answer)
        print(f"Bot: {answer}")
