import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import requests
from dotenv import load_dotenv
import os
load_dotenv()


API_KEY = os.getenv('HUGGING_FACE_API_KEY')

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Step 2: Chunk the text
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 3: Embed the text chunks
def embed_text_chunks(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings

# Step 4: Initialize Faiss index and store embeddings
def store_embeddings_in_faiss(embeddings):
    try:
        d = embeddings.shape[1]  # Dimension of the embeddings
        index = faiss.IndexFlatIP(d)  # Using IndexFlatIP for inner product similarity
        index.add(embeddings)
        print(f"Embeddings stored in Faiss index with dimension {d}")
        return index
    except Exception as e:
        print(f"Error storing embeddings in Faiss index: {e}")
        return None

# Step 5: Handle query and generate response
def generate_response(query, index, text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    print(f"Query Embedding: {query_embedding}")

    try:
        D, I = index.search(query_embedding.reshape(1, -1), 5)  # Search for top 5 most similar embeddings
        print(f"Results: {I}")

        retrieved_texts = [text_chunks[i] for i in I[0]]
        context = ' '.join(retrieved_texts)
        print(f"Context: {context}")

        # Use Hugging Face API for text generation
        try:
            headers = {
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {
                'inputs': context + "\n\n" + query,
                'max_length': 256
            }
            response = requests.post('https://api-inference.huggingface.co/models/gpt2', headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                generated_text = result[0]['generated_text']
                return generated_text
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return "An error occurred while generating the response."

        except Exception as e:
            print(f"Error generating response with Hugging Face API: {e}")
            return "An error occurred while generating the response."

    except Exception as e:
        print(f"Error querying Faiss index: {e}")
        return "An error occurred while querying Faiss index."

if __name__ == "__main__":
    pdf_path = 'samplePDF.pdf'
    
    # Extract and chunk text
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted Text: {text[:100]}")  # Print the first 100 characters of the extracted text
    text_chunks = chunk_text(text)
    print(f"First 3 Text Chunks: {text_chunks[:3]}")  # Print the first 3 chunks
    
    # Embed text chunks
    embeddings = embed_text_chunks(text_chunks)
    print(f"First 3 Embeddings: {embeddings[:3]}")  # Print the first 3 embeddings
    
    # Store embeddings in Faiss index
    index = store_embeddings_in_faiss(embeddings)
    
    if index is not None:
        # Query and generate response example
        query = "What is the greedy algorithm? Provide short one sentence definition."
        response = generate_response(query, index, text_chunks)
        print(f"Response: {response}")