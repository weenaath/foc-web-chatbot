import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_text_from_url(url):
    # Download the webpage
    response = requests.get(url)
    
    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove scripts and styles (if any)
    for s in soup(["script", "style"]):
        s.extract()
        
    # Extract visible text
    text = soup.get_text()
    
    # Clean: remove extra spaces and newlines
    text = " ".join(text.split())
    return text

def chunk_text(text, size=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# Load data
url = "https://computing.sjp.ac.lk/"
content = get_text_from_url(url)
chunks = chunk_text(content, size=200)

# Encode with Sentence-BERT
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

def get_answer(question):
    q_vec = model.encode([question])
    sims = cosine_similarity(q_vec, embeddings)
    idx = np.argmax(sims)
    return chunks[idx]

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("Answer:", get_answer(q))
