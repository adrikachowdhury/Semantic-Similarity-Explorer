# Semantic Similarity Explorer

**Project Type:** NLP Foundations / Linear Algebra

Live Demo: [Semantic Simlarity Explorer](https://semantic-similarity.streamlit.app/)

![App Demo](assets/demo.gif)

---

## 🧠 Project Overview

This project demonstrates a **document similarity engine built from scratch using linear algebra foundations**.

- Converts text documents into numeric vectors using **Bag-of-Words**
- Computes **similarity between documents** using:
  - **Dot Product**
  - **Cosine Similarity**
- Provides **interactive query search**
- Visualizes document alongside user query vectors in 2D using **Principal Component Analysis (PCA)**

This showcases foundational skills required for NLP, embeddings, and retrieval-based Generative AI systems.

---

## 📚 Key Concepts Implemented

- **Vector representation of documents**
- **Matrix multiplication for similarity computation**
- **Normalization**
- **Cosine similarity**
- **PCA for visualization**
- **Interactive web deployment using Streamlit**

---

## 💻 How to Run Locally

1. Clone the repo:

```bash
git clone https://github.com/<username>/semantic-similarity-explorer.git
cd semantic-similarity-explorer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run semantic-similarity-app.py
```

4. Open the local URL in your browser to interact with the app.

5. Optional- If you want to test this out on a notebook like the Google Colab, then try with `colab-version.ipynb`. N.B.- this notebook only contains the core analysis of the given documents (no user query input is present)

## 🔍 Usage

- View the list of documents given (you can modify them on your own)
- See vocabulary and document vectors
- Explore dot product and cosine similarity
- Enter your own query to see ranked results
- Visualize documents and user queries in 2D PCA space
- Plus point: Look into the thorough documentation of the code to understand each critical segment of it

## 📈 Learning Outcomes

This project demonstrates:

- Application of linear algebra in NLP
- Understanding vector spaces and similarity measures
- Building interactive data-driven applications
- Preparing for embedding-based and transformer-based NLP systems

## 💫 Credits
Feel free to explore the documentation, and please give **credit to the owner** when using content from this repo! 
Many thanks!🙌
