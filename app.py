from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)

# Global storage
chunks = []
vectorizer = None
tfidf_matrix = None

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

def split_text(text, chunk_size=500):
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

def create_tfidf(chunks_list):
    vect = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = vect.fit_transform(chunks_list)
    return vect, matrix

def get_relevant_chunks(query, vect, matrix, chunks_list, k=3):
    query_vec = vect.transform([query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    top_indices = similarities.argsort()[-k:][::-1]
    return [chunks_list[i] for i in top_indices]

def ask_gemini(api_key, question, context):
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    prompt = f"""Answer based ONLY on this context. If not found, say "I don't have enough information."

Context: {context}

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
    global chunks, vectorizer, tfidf_matrix
    
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
    
    vectorizer, tfidf_matrix = create_tfidf(all_chunks)
    chunks = all_chunks
    
    return jsonify({
        'success': True,
        'message': f'Processed {len(files)} files, {len(all_chunks)} chunks ready!'
    })

@app.route('/chat', methods=['POST'])
def chat():
    global chunks, vectorizer, tfidf_matrix
    
    if not chunks or vectorizer is None:
        return jsonify({'error': 'Please upload PDFs first'})
    
    data = request.get_json()
    question = data.get('question', '')
    api_key = data.get('api_key', os.getenv('GOOGLE_API_KEY', ''))
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    relevant = get_relevant_chunks(question, vectorizer, tfidf_matrix, chunks)
    context = "\n\n".join(relevant)
    answer = ask_gemini(api_key, question, context)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5000)