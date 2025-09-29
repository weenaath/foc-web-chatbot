# bot_improved.py
import time
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
import os
import pickle
from tqdm import tqdm
import numpy as np

# Embedding model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import faiss, else fallback
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ---------------------------
# Configuration (tweakable)
# ---------------------------
SEED_URL = "https://computing.sjp.ac.lk/"    
MAX_PAGES = 200          # how many pages to crawl at most
CRAWL_DELAY = 0.8        # seconds between requests (be polite)
CHUNK_SENTENCES = 6      # target sentences per chunk
CHUNK_OVERLAP = 2        # overlapping sentences between chunks
TOP_K = 5                # how many top chunks to retrieve
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------
# Utilities: robots check (basic)
# ---------------------------
from urllib.robotparser import RobotFileParser
def allowed_by_robots(url, user_agent="*"):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # if robots unreachable, be conservative but allow for now

# ---------------------------
# Crawl multiple pages (domain-limited BFS)
# ---------------------------
def crawl_site(seed_url, max_pages=MAX_PAGES):
    parsed_seed = urlparse(seed_url)
    base_netloc = parsed_seed.netloc

    to_visit = {seed_url}
    visited = set()
    pages = []

    headers = {"User-Agent": "FacultyQA-Bot/1.0 (+https://computing.sjp.ac.lk/)"}

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited: 
            continue
        if urlparse(url).netloc != base_netloc:
            continue
        if not allowed_by_robots(url):
            print("Skipping (robots):", url)
            visited.add(url)
            continue
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Extract page title and main text (prefer <main>, <article>)
            title = (soup.title.string.strip() if soup.title else url)
            main = soup.find("main") or soup.find("article")
            if main:
                texts = main.get_text(separator=" ")
            else:
                # as fallback, exclude nav/footer/script/style
                for s in soup(["script", "style", "nav", "footer", "header"]):
                    s.decompose()
                texts = soup.get_text(separator=" ")

            text = " ".join(texts.split())
            pages.append({"url": url, "title": title, "text": text})
            visited.add(url)

            # Find internal links and add to to_visit
            for a in soup.find_all("a", href=True):
                href = a['href'].strip()
                # build absolute
                href = urljoin(url, href)
                parsed = urlparse(href)
                # only same domain, http/https, remove fragments
                if parsed.scheme not in ("http", "https"):
                    continue
                href = href.split("#")[0]
                if parsed.netloc == base_netloc and href not in visited and href not in to_visit:
                    to_visit.add(href)

            time.sleep(CRAWL_DELAY)
        except Exception as e:
            # don't crash the crawl on a single page error
            print("Error fetching", url, ":", str(e))
            visited.add(url)
            continue

    return pages

# ---------------------------
# Text -> sentences -> chunking (with overlap)
# ---------------------------
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text):
    # simple sentence splitter (no external NLTK dependency)
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents

def chunk_sentences_from_text(text, chunk_size=CHUNK_SENTENCES, overlap=CHUNK_OVERLAP):
    sents = split_sentences(text)
    chunks = []
    i = 0
    while i < len(sents):
        chunk_sents = sents[i:i+chunk_size]
        chunk_text = " ".join(chunk_sents)
        chunks.append(chunk_text)
        if i + chunk_size >= len(sents):
            break
        i += (chunk_size - overlap)
    return chunks

def pages_to_chunks(pages):
    chunk_records = []  # each item: {id, url, title, chunk_text}
    cid = 0
    for p in pages:
        chunks = chunk_sentences_from_text(p['text'], CHUNK_SENTENCES, CHUNK_OVERLAP)
        for c in chunks:
            chunk_records.append({"id": cid, "url": p['url'], "title": p['title'], "text": c})
            cid += 1
    return chunk_records

# ---------------------------
# Build or load embeddings + index
# ---------------------------
class Retriever:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.embeddings = None

    def build(self, chunk_records, use_faiss=_HAS_FAISS):
        texts = [r['text'] for r in chunk_records]
        # encode -> numpy array
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # normalize for cosine with dot-product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms==0] = 1e-9
        emb = emb / norms

        self.embeddings = emb
        self.metadata = chunk_records

        if use_faiss:
            dim = emb.shape[1]
            # inner-product on normalized vectors == cosine similarity
            index = faiss.IndexFlatIP(dim)
            index.add(emb.astype(np.float32))
            self.index = index
        else:
            self.index = None

    def save(self, path="index_data"):
        os.makedirs(path, exist_ok=True)
        # embeddings
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        if _HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "faiss.idx"))

    def load(self, path="index_data"):
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        if _HAS_FAISS:
            try:
                self.index = faiss.read_index(os.path.join(path, "faiss.idx"))
            except Exception:
                self.index = None

    def query(self, question, top_k=TOP_K):
        q_emb = self.model.encode([question], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        if self.index is not None:
            # use faiss
            D, I = self.index.search(q_emb.astype(np.float32), top_k)
            scores = D[0].tolist()
            idxs = I[0].tolist()
        else:
            # fallback: cosine with numpy
            sims = (self.embeddings @ q_emb.T).squeeze()  # dot with normalized vectors
            idxs = np.argsort(-sims)[:top_k].tolist()
            scores = sims[idxs].tolist()
        results = []
        for i, s in zip(idxs, scores):
            results.append({"score": float(s), "meta": self.metadata[i]})
        return results

# ---------------------------
# Answer synthesis: pick best sentences from top chunks
# ---------------------------
def synthesize_answer(question, top_chunks, retriever, max_sentences=3):
    # collect sentences from top chunks
    all_sents = []
    sent_origin = []
    for r in top_chunks:
        txt = r['meta']['text']
        sents = split_sentences(txt)
        for s in sents:
            all_sents.append(s)
            sent_origin.append({"url": r['meta']['url'], "title": r['meta']['title']})

    if not all_sents:
        return "No text available to answer."

    # encode sentences in batch (efficient)
    sent_embs = retriever.model.encode(all_sents, convert_to_numpy=True)
    # normalize
    sent_embs = sent_embs / (np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-9)
    q_emb = retriever.model.encode([question], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

    sims = (sent_embs @ q_emb.T).squeeze()
    top_idx = np.argsort(-sims)[:max_sentences]

    chosen = []
    sources = []
    for i in top_idx:
        chosen.append(all_sents[int(i)])
        sources.append(sent_origin[int(i)])
    answer = " ".join(chosen)
    # compact answer + sources + confidence
    source_lines = []
    for s in sources:
        source_lines.append(f"{s['title']} - {s['url']}")
    return {
        "answer": answer,
        "sources": list(dict.fromkeys(source_lines)),  # unique
        "confidence": float(np.mean(sims[top_idx]))
    }

# ---------------------------
# Putting it all together
# ---------------------------
def build_and_save_index(seed_url=SEED_URL):
    print("Crawling site...")
    pages = crawl_site(seed_url, max_pages=MAX_PAGES)
    print(f"Pages crawled: {len(pages)}")
    print("Converting pages to chunks...")
    chunks = pages_to_chunks(pages)
    print("Number of chunks:", len(chunks))

    print("Building retriever and embeddings...")
    r = Retriever()
    r.build(chunks)
    r.save()
    print("Index saved to ./index_data")
    return r

def load_index(path="index_data"):
    r = Retriever()
    r.load(path)
    return r

# Small interactive demo
if __name__ == "__main__":
    if not os.path.exists("index_data"):
        retriever = build_and_save_index(SEED_URL)
    else:
        retriever = load_index("index_data")
        print("Index loaded. chunks:", len(retriever.metadata))

    print("Ready. Ask questions (type 'exit'):")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit","quit"):
            break
        top = retriever.query(q, top_k=TOP_K)
        synth = synthesize_answer(q, top, retriever, max_sentences=3)
        print("\n--- Answer ---\n")
        print(synth['answer'])
        print("\nSources:")
        for s in synth['sources']:
            print("-", s)
        print(f"\nConfidence (mean cosine): {synth['confidence']:.3f}\n")
