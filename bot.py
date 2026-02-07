import requests
import json
from data_processing import clean_text, make_chunks
from retrieval import setup_db, funnel_retrieve

OPENROUTER_KEY = "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"

def ask_bot(query, collection):
    context = "\n---\n".join(funnel_retrieve(query, collection))
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        data=json.dumps({
            "model": "google/gemini-2.0-flash-001",
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    return response.json()['choices'][0]['message']['content']