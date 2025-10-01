# 📚 Faculty Chatbot – University of Sri Jayewardenepura (FoC)

This project is a **chatbot powered by AI** that can answer questions about the **Faculty of Computing (FoC), University of Sri Jayewardenepura**.  
It uses the **Groq API (LLaMA models)** along with a **cleaned knowledge base** extracted from the official faculty website.  

---

## 🚀 Features
- ✅ Answers common questions about courses, departments, and staff.  
- ✅ Knowledge base cleaned and summarized from the official FoC website.  
- ✅ Runs locally in a simple command-line interface.  
- ✅ Can be extended for deployment on a web app or faculty site.  

---

## 🛠️ Tech Stack
- **Python 3.8+**  
- [Groq API](https://groq.com/) (OpenAI-compatible)  
- `requests`, `python-dotenv` for API and environment handling  
- `beautifulsoup4` (used for scraping, optional)  

---

## 📂 Project Structure
foc-web-chatbot/
│── bot.py # Main chatbot script (Groq API + knowledge base)
│── scraper.py # Scraper for faculty website (optional)
│── faculty_content.txt # Cleaned knowledge base
│── .env # API key stored here
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── venv/ # Virtual environment (not pushed to GitHub)

## ⚙️ Setup & Installation

1. **Clone the repo**
   git clone https://github.com/yourusername/foc-web-chatbot.git
   cd foc-web-chatbot

2. **Create & activate virtual environment**
   python -m venv venv
   venv\Scripts\activate   # on Windows CMD
   # OR
   .\venv\Scripts\Activate # on PowerShell

3. **Install dependencies**
   pip install -r requirements.txt

3. **Set up API key**
   GROQ_API_KEY=your_api_key_here # I used Groq API for this

4. **▶️ Run the Chatbot**
   python bot.py

   **You should see:**
   🤖 Faculty of Computing Chatbot is running! Type 'quit' to exit.
   You: