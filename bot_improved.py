import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- Configuration ----------------
SEED_URL = "https://computing.sjp.ac.lk/"
# Whitelist: only crawl pages containing any of these substrings
WHITELIST_KEYWORDS = ["courses", "computing.sjp.ac.lk", "staff", "departments", 
                      "bachelor-of-computing-honours", "contact"]
MAX_PAGES = 50
CRAWL_DELAY = 0.5

CHUNK_SENTENCES = 2
CHUNK_OVERLAP = 1
TOP_K = 8
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------- Helpers ----------------

def allowed_domain(url):
    # allow only your domain
    p = urlparse(url)
    return "computing.sjp.ac.lk" in p.netloc

def url_whitelisted(url):
    # decide whether to crawl this url
    for kw in WHITELIST_KEYWORDS:
        if kw in url:
            return True
    return False

def fetch_page_text(url):
    try:
        resp = requests.get(url, timeout=7)
        resp.raise_for_status()
    except Exception as e:
        print("Fetch error:", url, e)
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    # remove scripts/styles
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    # extract headings
    heading_text = " ".join(h.get_text(" ", strip=True) for h in soup.find_all(["h1","h2","h3"]))
    body = soup.get_text(separator=" ")
    # combine headings + body
    text = heading_text + " " + body
    text = " ".join(text.split())
    title = soup.title.string.strip() if soup.title else url
    return {"url": url, "title": title, "text": text}

def crawl_whitelist(start_url):
    to_visit = {start_url}
    visited = set()
    pages = []
    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop()
        if url in visited:
            continue
        if not allowed_domain(url):
            visited.add(url)
            continue
        page = fetch_page_text(url)
        visited.add(url)
        if page is None:
            continue
        # only keep white-listed pages
        if url_whitelisted(url):
            pages.append(page)
        # find links
        soup = BeautifulSoup(requests.get(url).text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            href = href.split("#")[0]
            if href not in visited and allowed_domain(href):
                to_visit.add(href)
        time.sleep(CRAWL_DELAY)
    return pages

# Sentence splitting + chunking with overlap
_SENT_RE = re.compile(r'(?<=[.!?])\s+')
def split_sentences(text):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

def chunk_texts(pages):
    chunks = []
    cid = 0
    for p in pages:
        sents = split_sentences(p["text"])
        i = 0
        n = len(sents)
        while i < n:
            block = sents[i : i + CHUNK_SENTENCES]
            if block:
                chunk = " ".join(block)
                chunks.append({"id": cid, "url": p["url"], "title": p["title"], "text": chunk})
                cid += 1
            if i + CHUNK_SENTENCES >= n:
                break
            i += (CHUNK_SENTENCES - CHUNK_OVERLAP)
    return chunks

class Retriever:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.meta = []
    def build(self, chunks):
        texts = [c["text"] for c in chunks]
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-9)
        self.embeddings = emb
        self.meta = chunks
    def save(self, path="refined_index"):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "emb.npy"), self.embeddings)
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(self.meta, f)
    def load(self, path="refined_index"):
        self.embeddings = np.load(os.path.join(path, "emb.npy"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
    def query(self, question, top_k=TOP_K):
        qv = self.model.encode([question], convert_to_numpy=True)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        sims = (self.embeddings @ qv.T).squeeze()
        idxs = np.argsort(-sims)[:top_k]
        return [{"score": float(sims[i]), "meta": self.meta[i]} for i in idxs]

def synthesize(question, candidates):
    # find sentences within those chunks
    sents = []
    sent_meta = []
    for c in candidates:
        ts = split_sentences(c["meta"]["text"])
        for s in ts:
            sents.append(s)
            sent_meta.append({"url": c["meta"]["url"], "title": c["meta"]["title"]})
    if not sents:
        return {"answer": "Sorry, I couldn't find information.", "sources": [], "confidence": 0.0}
    sent_embs = Retriever().model.encode(sents, convert_to_numpy=True)
    # normalization
    sent_embs = sent_embs / (np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-9)
    qv = Retriever().model.encode([question], convert_to_numpy=True)
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    sims = (sent_embs @ qv.T).squeeze()
    top = np.argsort(-sims)[:2]  # pick up to 2 sentences
    answer = " ".join(sents[i] for i in top)
    sources = []
    for i in top:
        sources.append(f"{sent_meta[i]['title']} - {sent_meta[i]['url']}")
    return {"answer": answer, "sources": list(dict.fromkeys(sources)), "confidence": float(np.mean(sims[top]))}

# Main
if __name__ == "__main__":
    # Build or load
    if not os.path.exists("refined_index"):
        print("Crawling and building index …")
        pages = crawl_whitelist(SEED_URL)
        print("Pages:", len(pages))
        chunks = chunk_texts(pages)
        print("Chunks:", len(chunks))
        retr = Retriever()
        retr.build(chunks)
        retr.save()
    else:
        retr = Retriever()
        retr.load()
        print("Index loaded, chunks:", len(retr.meta))

    print("Ready to answer! (type 'exit')")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        cands = retr.query(q, TOP_K)
        resp = synthesize(q, cands)
        print("\n--- Answer ---\n")
        print(resp["answer"])
        print("\nSources:")
        for s in resp["sources"]:
            print("-", s)
        print("Confidence:", resp["confidence"])
        print("\n----------------\n")
