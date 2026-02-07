import re
import spacy
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text) # Remove stage directions
    text = re.sub(r'\s+', ' ', text)    # Remove extra whitespace
    return text.strip()

def make_chunks(text, sentences_per_chunk=5):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return [" ".join(sentences[i : i + sentences_per_chunk]) 
            for i in range(0, len(sentences), sentences_per_chunk)]