from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load models and data
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

with open('X_reduced.pkl', 'rb') as f:
    X_reduced = pickle.load(f)

# Load the original documents
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Transform the query
    query_vec = vectorizer.transform([query])
    query_vec_reduced = svd_model.transform(query_vec)

    # Compute cosine similarities
    similarities = cosine_similarity(query_vec_reduced, X_reduced)[0]

    # Get top 5 documents
    top_indices = similarities.argsort()[-5:][::-1]
    top_scores = similarities[top_indices]
    top_docs = [documents[i] for i in top_indices]

    # Prepare the results
    results = []
    for idx in range(5):
        results.append({
            'document': top_docs[idx][:200] + '...',  # Preview first 200 chars
            'score': round(float(top_scores[idx]), 4)
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=3000)
