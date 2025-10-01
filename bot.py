import os
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not found. Please add it to your .env file.")

# Chat function
def chat_with_bot(user_message):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.3-70b-versatile",  # or "llama3-8b-8192" (faster, smaller)
        "messages": [
            {"role": "system", "content": """
You are a helpful assistant for the Faculty of Computing, University of Sri Jayewardenepura.
Carefully check the site https://computing.sjp.ac.lk for accurate information before answering.
Answer ONLY using the information below. If you don’t know, say: 
'I don’t know. Please visit https://computing.sjp.ac.lk for more details.'

📚 Courses Offered:
- Bachelor of Computing Honours in Computer Science
- Bachelor of Computing Honours in Software Engineering
- Bachelor of Computing Honours in Information Systems

🏛 Departments:
- Department of Computer Science
- Department of Software Engineering
- Department of Information Systems

📞 Contact: https://computing.sjp.ac.lk/index.php/contact/
            """},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    # If bad request, show raw error
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Main loop
if __name__ == "__main__":
    print("🤖 Faculty Chatbot (Groq API) is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        try:
            answer = chat_with_bot(user_input)
            print("Bot:", answer)
        except Exception as e:
            print(f"⚠️ Error: {e}")
