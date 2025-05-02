from flask import Flask, request, jsonify
from flask_cors import CORS
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# ------------------- CONFIG -------------------
PDF_FOLDER_PATH = r"C:\Users\desai\Downloads\Extension\extension\textbooks"

# ------------------- PDF CHUNK CACHE -------------------

def extract_chunks_from_pdfs(folder_path):
    chunks = []
    sources = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                try:
                    pdf_path = os.path.join(root, file)
                    doc = fitz.open(pdf_path)
                    for i, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            chunks.append(text.strip())
                            sources.append(f"{pdf_path} - Page {i+1}")
                except Exception as e:
                    print(f"[!] Failed to read {file}: {e}")
    return chunks, sources

PDF_CHUNKS, PDF_SOURCES = extract_chunks_from_pdfs(PDF_FOLDER_PATH)

# ------------------- UTILITY FUNCTIONS -------------------

def extract_keywords_from_question(question):
    words = re.findall(r'\b\w+\b', question.lower())
    stopwords = {
        'what', 'is', 'the', 'of', 'in', 'a', 'an', 'on', 'for', 'how', 'why', 'which',
        'to', 'by', 'from', 'are', 'can', 'be', 'it', 'that', 'this', 'and', 'with', 'as'
    }
    return [w for w in words if w not in stopwords and len(w) > 2 and not w.isdigit()]

def search_fallback_pdfs(question, top_n=10):
    if not PDF_CHUNKS:
        return None

    keywords = extract_keywords_from_question(question)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85).fit_transform(PDF_CHUNKS + [question])
    cosine_sim = cosine_similarity(vectorizer[-1:], vectorizer[:-1]).flatten()

    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    results = []

    for idx in top_indices:
        match_text = PDF_CHUNKS[idx]
        if any(kw in match_text.lower() for kw in keywords):
            results.append({
                'source': PDF_SOURCES[idx],
                'answer': match_text[:1000].strip()
            })

    return results if results else None

def clean_html_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def fetch_google_content(question, keywords):
    try:
        search_results = list(search(question, num_results=5))
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/90.0.4430.93 Safari/537.36"
        }

        for url in search_results:
            try:
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                texts = []

                for tag in soup.find_all(['p', 'li']):
                    raw_text = tag.get_text()
                    cleaned = clean_html_text(raw_text)
                    if 100 < len(cleaned) < 800 and any(kw in cleaned.lower() for kw in keywords):
                        texts.append(cleaned)

                if texts:
                    return [{
                        'source': url,
                        'answer': "\n\n".join(texts[:3])
                    }]
            except Exception as e:
                print(f"[!] Skipped {url}: {e}")
    except Exception as e:
        print(f"[!] Google Search Error: {e}")
    return None

def fetch_duckduckgo_content(question, keywords):
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(question)}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for a in soup.select('a.result__a'):
            href = a.get('href')
            if href:
                results.append({'source': href, 'answer': a.get_text(strip=True)})
        return results[:5] if results else None
    except Exception as e:
        print(f"[!] DuckDuckGo Error: {e}")
    return None

# ------------------- API ROUTES -------------------

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to Custom BPharm QA API. Use POST /answer to ask questions."})

@app.route('/answer', methods=['POST'])
def answer():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    keywords = extract_keywords_from_question(question)

    with ThreadPoolExecutor(max_workers=3) as executor:
        google_future = executor.submit(fetch_google_content, question, keywords)
        duck_future = executor.submit(fetch_duckduckgo_content, question, keywords)
        pdf_future = executor.submit(search_fallback_pdfs, question)

        google_results = google_future.result()
        duck_results = duck_future.result()
        pdf_results = pdf_future.result()

    return jsonify({
        'google_results': google_results or [{'error': 'No good answer from Google'}],
        'duckduckgo_results': duck_results or [{'error': 'No good answer from DuckDuckGo'}],
        'pdf_results': pdf_results or [{'error': 'No good match in local PDFs'}]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
