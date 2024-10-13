from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Create the term-document matrix
X = vectorizer.fit_transform(documents)

# Set the number of components for dimensionality reduction
n_components = 100

# Apply SVD
svd_model = TruncatedSVD(n_components=n_components)
X_reduced = svd_model.fit_transform(X)

# Save the TF-IDF vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the SVD model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd_model, f)

# Save the reduced document vectors
with open('X_reduced.pkl', 'wb') as f:
    pickle.dump(X_reduced, f)
