import requests
from bs4 import BeautifulSoup
import re

# Pages to crawl
PAGES = {
    "home": "https://computing.sjp.ac.lk/",
    "staff": "https://computing.sjp.ac.lk/index.php/staff/",
    "departments": "https://computing.sjp.ac.lk/index.php/departments/",
    "courses": "https://computing.sjp.ac.lk/index.php/courses/",
    "cs_degree": "https://computing.sjp.ac.lk/index.php/bachelor-of-computing-honours-in-computer-science/",
    "se_degree": "https://computing.sjp.ac.lk/index.php/bachelor-of-computing-honours-in-software-engineering/",
    "is_degree": "https://computing.sjp.ac.lk/index.php/bachelor-of-computing-honours-in-information-systems/",
    "contact": "https://computing.sjp.ac.lk/index.php/contact/",
}

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def scrape_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    # Get only visible text
    paragraphs = soup.find_all(["p", "h1", "h2", "h3", "li"])
    return clean_text(" ".join([p.get_text() for p in paragraphs]))

if __name__ == "__main__":
    with open("faculty_chunks.txt", "w", encoding="utf-8") as f:
        for name, url in PAGES.items():
            content = scrape_page(url)
            f.write(f"--- {name} ---\n{content}\n\n")
    print("✅ Faculty content saved into faculty_chunks.txt")
