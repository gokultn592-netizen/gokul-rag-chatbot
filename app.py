from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import math
import re
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)

# Global storage
chunks = []
tfidf_vectors = []
idf = {}
vocab = {}

def extract_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        return f"Error: {str(e)}"

def split_text(text, chunk_size=1000):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks_list = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += sent + " "
        else:
            if current:
                chunks_list.append(current.strip())
            current = sent + " "
    if current:
        chunks_list.append(current.strip())
    return chunks_list

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()

def compute_tfidf(docs):
    global vocab, idf, tfidf_vectors
    
    vocab = {}
    doc_freq = {}
    for doc in docs:
        tokens = set(tokenize(doc))
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    
    vocab = {token: idx for idx, token in enumerate(doc_freq.keys())}
    N = len(docs)
    idf = {token: math.log((N + 1) / (df + 1)) + 1 for token, df in doc_freq.items()}
    
    tfidf_vectors = []
    for doc in docs:
        tokens = tokenize(doc)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        
        vec = {}
        for token, idx in vocab.items():
            tf_val = tf.get(token, 0) / len(tokens) if tokens else 0
            vec[idx] = tf_val * idf.get(token, 0)
        tfidf_vectors.append(vec)
    
    return tfidf_vectors

def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def get_relevant_chunks(query, chunks_list, k=3):
    global vocab, idf, tfidf_vectors
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Check if it's a "summary" or "about" question
    summary_keywords = ['about', 'summarize', 'summary', 'what is this', 'main topic', 'overview', 'what is this pdf', 'what is this document']
    is_summary_question = any(kw in query_lower for kw in summary_keywords)
    
    # For summary questions, return first chunks (beginning of doc)
    if is_summary_question:
        return chunks_list[:k]
    
    # For specific questions, use TF-IDF + cosine similarity
    q_tokens = tokenize(query)
    q_tf = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
    
    q_vec = {}
    for token, idx in vocab.items():
        tf_val = q_tf.get(token, 0) / len(q_tokens) if q_tokens else 0
        q_vec[idx] = tf_val * idf.get(token, 0)
    
    scores = []
    for i, doc_vec in enumerate(tfidf_vectors):
        score = cosine_similarity(q_vec, doc_vec)
        scores.append((score, i))
    
    scores.sort(reverse=True)
    
    # If no good matches, return first chunks as fallback
    if not scores or scores[0][0] == 0:
        return chunks_list[:k]
    
    return [chunks_list[i] for _, i in scores[:k]]

def ask_gemini(api_key, question, context):
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Detect summary questions
    summary_keywords = ['about', 'summarize', 'summary', 'what is this', 'main topic', 'overview', 'what is this pdf', 'what is this document']
    is_summary = any(kw in question.lower() for kw in summary_keywords)
    
    if is_summary:
        prompt = f"""Based on the following document excerpts, provide a brief summary of what this document is about. Be concise but informative.

Document excerpts:
{context}

Summary:"""
    else:
        prompt = f"""Answer the question based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global chunks, tfidf_vectors, idf, vocab
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files')
    api_key = request.form.get('api_key', os.getenv('GOOGLE_API_KEY', ''))
    
    if not api_key:
        return jsonify({'error': 'No API key provided'})
    
    all_chunks = []
    for file in files:
        if file.filename.endswith('.pdf'):
            text = extract_text(file)
            if text and not text.startswith('Error'):
                file_chunks = split_text(text)
                all_chunks.extend(file_chunks)
    
    if not all_chunks:
        return jsonify({'error': 'No text extracted from PDFs'})
    
    compute_tfidf(all_chunks)
    chunks = all_chunks
    
    return jsonify({
        'success': True,
        'message': f'Processed {len(files)} files, {len(all_chunks)} chunks ready!'
    })

@app.route('/chat', methods=['POST'])
def chat():
    global chunks, tfidf_vectors
    
    if not chunks or not tfidf_vectors:
        return jsonify({'error': 'Please upload PDFs first'})
    
    data = request.get_json()
    question = data.get('question', '')
    api_key = data.get('api_key', os.getenv('GOOGLE_API_KEY', ''))
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    relevant = get_relevant_chunks(question, chunks)
    context = "\n\n".join(relevant)
    answer = ask_gemini(api_key, question, context)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
