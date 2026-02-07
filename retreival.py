import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import numpy as np

# Setup
client = chromadb.PersistentClient(path="./chroma_db") # Saves DB locally
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def setup_db(chunks):
    collection = client.get_or_create_collection(name="shakespeare", embedding_function=emb_fn)
    collection.add(documents=chunks, ids=[f"id_{i}" for i in range(len(chunks))])
    return collection

def funnel_retrieve(query, collection, broad_k=50, precise_k=5):
    results = collection.query(query_texts=[query], n_results=broad_k)
    candidates = results['documents'][0]
    
    # Rerank
    scores = reranker.predict([[query, c] for c in candidates])
    top_indices = np.argsort(scores)[::-1][:precise_k]
    return [candidates[i] for i in top_indices]