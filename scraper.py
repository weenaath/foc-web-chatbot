import requests
from bs4 import BeautifulSoup

# Faculty website important pages
pages = {
    "home": "https://computing.sjp.ac.lk/",
    "staff": "https://computing.sjp.ac.lk/index.php/staff/",
    "departments": "https://computing.sjp.ac.lk/index.php/departments/",
    "courses": "https://computing.sjp.ac.lk/index.php/courses/",
    "cs_degree": "https://computing.sjp.ac.lk/index.php/bachelor-of-computing-honours-in-computer-science/",
    "se_degree": "https://computing.sjp.ac.lk/index.php/bachelor-of-computing-honours-in-software-engineering/",
    "is_degree": "https://computing.sjp.ac.lk/index.php/bachelor-of-computing-honours-in-information-systems/",
    "contact": "https://computing.sjp.ac.lk/index.php/contact/",
}

def scrape_page(url):
    """Fetch text from a single page"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, and navs
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.extract()

    text = soup.get_text(separator="\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def scrape_all():
    """Scrape all pages into a dictionary"""
    data = {}
    for name, url in pages.items():
        print(f"Scraping {name}...")
        data[name] = scrape_page(url)
    return data

if __name__ == "__main__":
    content = scrape_all()
    with open("faculty_content.txt", "w", encoding="utf-8") as f:
        for page, text in content.items():
            f.write(f"\n\n--- {page.upper()} ---\n{text}\n")
    print("✅ Faculty website content saved to faculty_content.txt")
