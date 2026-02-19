import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # Converts text → numeric vectors (Bag-of-Words)
from sklearn.decomposition import PCA #Reduces high-dimensional vectors → 2D for visualization
import matplotlib.pyplot as plt #To draw the scatter plot

st.title("📄 Document Similarity Explorer")

"""
In Streamlit, we use st.title(), which is 
similar to Python's print() function
"""

# -----------------------------
# Step 1: Define Documents
# -----------------------------

#Python list
"""
These are sample documents. You can modify
by adding any documents of your own
"""
documents = [
    "Climate change affects global temperature.",
    "Global warming impacts weather patterns.",
    "Artificial intelligence transforms industries.",
    "NLP is a subset of Artificial Intelligence.",
    "Renewable energy reduces climate change."
]

st.subheader("📚 Documents") #smaller heading

#i- index; doc- document text
for i, doc in enumerate(documents):
    st.write(f"Document {i}: {doc}")

# -----------------------------
# Step 2: Vectorization
# -----------------------------

"""
fit.transform()- learn vocab + converts docs into matrix
"""
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
X = X.toarray() #(number_of_docs, vocabulary_size)

st.subheader("🧠 Vocabulary")
st.write("These are the vocabularies that are extracted from the documents")
st.write(vectorizer.get_feature_names_out())

st.subheader("📊 Document Vectors")
st.write("Each row = document, and each column = word count")
st.write(X) #full document-term matrix

# -----------------------------
# Step 3: Similarity Computation
# -----------------------------

similarity_matrix = np.dot(X, X.T) #result- each cell (i,j) = dot product between doc i and j
st.subheader("🔁 Raw Dot Product Similarity Matrix")
st.write(similarity_matrix) #Streamlit automatically formats NumPy arrays nicely

"""
We compute the vector length for each document. If X shape = (5, V),
then axis=1 → compute row-wise. So we get 5 norms.
Without keepdims=True, shape would be (5,)
With keepdims=True → (5, 1)
"""
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_normalized = X / norms #broadcasting in action. Each document vector now has length = 1
cosine_similarity = np.dot(X_normalized, X_normalized.T) #Dot product of normalized vectors = cosine similarity

st.subheader("📐 Cosine Similarity Matrix")
st.write("It shows how similar two vectors are based on their direction (angle), ignoring their length or magnitude")
st.write(cosine_similarity)

# -----------------------------
# Step 4: Query Search
# -----------------------------

st.subheader("🔍 Search Query")
query = st.text_input("Enter a search query:")

#Only run search logic if user actually typed something
if query:

    """
    We don't use fit.transform() here, because we
    must use the SAME vocab learned earlier.

    transform() ensures same feature space
    """
    query_vec = vectorizer.transform([query]).toarray()

    """
    If query contains no known vocabulary
    words, the vector becomes all zeros
    """
    if np.linalg.norm(query_vec) == 0:
        st.warning("Query contains words not in vocabulary.")
    else:
        query_vec = query_vec / np.linalg.norm(query_vec)
        similarities = np.dot(X_normalized, query_vec.T).flatten()
        ranked_indices = np.argsort(similarities)[::-1]

        st.subheader("📌 Ranked Results")
        for idx in ranked_indices:
            st.write(f"Score: {similarities[idx]:.3f} | {documents[idx]}")

# -----------------------------------------------------
# Step 5: Dynamic PCA Visualization (Documents + Query)
# -----------------------------------------------------

st.subheader("📉 2D Visualization (Documents + User Query)")

if query and np.linalg.norm(query_vec) != 0:
    
    # Combine documents and query into one matrix
    combined = np.vstack([X, query_vec])

    # Apply PCA on combined matrix
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)

    # Split back
    docs_2d = combined_2d[:-1]
    query_2d = combined_2d[-1]

    fig, ax = plt.subplots()

    # Plot documents
    ax.scatter(docs_2d[:,0], docs_2d[:,1])
    
    for i in range(len(documents)):
        ax.annotate(f"Doc{i}", (docs_2d[i,0], docs_2d[i,1]))

    # Plot query in red
    ax.scatter(query_2d[0], query_2d[1])
    ax.annotate("Query", (query_2d[0], query_2d[1]))

    ax.set_title("Document Space with Query (PCA Projection)")
    st.pyplot(fig)

else:
    st.info("Enter a valid query to visualize its position in vector space.")
