import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("📄 Document Similarity Explorer")

# -----------------------------
# Step 1: Define Documents
# -----------------------------

documents = [
    "Climate change affects global temperature",
    "Global warming impacts weather patterns",
    "Artificial intelligence transforms industries",
    "NLP is a subset of Artificial intelligence",
    "Renewable energy reduces climate change"
]

st.subheader("📚 Documents")
for i, doc in enumerate(documents):
    st.write(f"Doc {i}: {doc}")

# -----------------------------
# Step 2: Vectorization
# -----------------------------

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
X = X.toarray()

st.subheader("🧠 Vocabulary")
st.write(vectorizer.get_feature_names_out())

st.subheader("📊 Document Vectors")
st.write(X)

# -----------------------------
# Step 3: Similarity Computation
# -----------------------------

similarity_matrix = np.dot(X, X.T)
st.subheader("🔁 Raw Dot Product Similarity Matrix")
st.write(similarity_matrix)

norms = np.linalg.norm(X, axis=1, keepdims=True)
X_normalized = X / norms
cosine_similarity = np.dot(X_normalized, X_normalized.T)

st.subheader("📐 Cosine Similarity Matrix")
st.write(cosine_similarity)

# -----------------------------
# Step 4: Query Search
# -----------------------------

st.subheader("🔍 Search Query")
query = st.text_input("Enter a search query:")

if query:
    query_vec = vectorizer.transform([query]).toarray()
    
    if np.linalg.norm(query_vec) == 0:
        st.warning("Query contains words not in vocabulary.")
    else:
        query_vec = query_vec / np.linalg.norm(query_vec)
        similarities = np.dot(X_normalized, query_vec.T).flatten()
        ranked_indices = np.argsort(similarities)[::-1]

        st.subheader("📌 Ranked Results")
        for idx in ranked_indices:
            st.write(f"Score: {similarities[idx]:.3f} | {documents[idx]}")

# -----------------------------
# Step 5: PCA Visualization
# -----------------------------

st.subheader("📉 2D Visualization of Document Vector Space")

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

fig, ax = plt.subplots()
ax.scatter(X_2d[:,0], X_2d[:,1])

for i in range(len(documents)):
    ax.annotate(f"Doc{i}", (X_2d[i,0], X_2d[i,1]))

ax.set_title("Document Vector Space (PCA)")
st.pyplot(fig)