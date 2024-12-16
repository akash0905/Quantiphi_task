from transformers import pipeline
import numpy as np
import torch
import faiss
import transformers

transformers.logging.set_verbosity_error()

try:
    faiss_index = faiss.read_index("faiss_index.bin")
    metadata_store = np.load("metadata_store.npy", allow_pickle=True)
    print("FAISS index and metadata loaded successfully.")
except Exception as e:
    faiss_index, metadata_store = None, None
    print(f"Error loading FAISS index or metadata: {e}")

try:
    from sentence_transformers import SentenceTransformer
    model_instance = SentenceTransformer('all-MiniLM-L6-v2')
    model_instance = model_instance.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    model_instance = None
    print(f"Error loading SentenceTransformer model: {e}")

try:
    llm_pipeline = pipeline(model="tiiuae/falcon-7b-instruct", device=0)
    print("LLM pipeline loaded successfully.")
except Exception as e:
    llm_pipeline = None
    print(f"Error loading LLM pipeline: {e}")

def validate_initialization(faiss_index, metadata, model, llm_pipeline):
    if not llm_pipeline:
        return "Error: LLM pipeline is not initialized."
    if not faiss_index:
        return "Error: FAISS index is not initialized."
    if not metadata.any():
        return "Error: Metadata store is not initialized."
    if not model:
        return "Error: Model instance for embedding is not initialized."
    return None

def retrieve_contexts(query_text, faiss_index, metadata, model, top_k=2):
    query_embedding = model.encode([query_text])
    distances, indices = faiss_index.search(np.array(query_embedding, dtype="float32"), top_k)
    return [metadata[idx]["text"] for idx, dist in zip(indices[0], distances[0]) if idx != -1]

def generate_response(query_text, contexts, llm_pipeline):
    if not contexts:
        return "Not Available"

    combined_context = "\n\n".join(contexts)
    full_prompt = f"Answer the following question based on the provided context:\n\n{combined_context}\n\nQuestion: {query_text}\n\nAnswer:"

    try:
        response = llm_pipeline(full_prompt, max_new_tokens=200, truncation=True)
        return response[0]['generated_text'].split("Answer:")[1].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error: Unable to generate response."

def query_and_respond(prompt, query_text, faiss_index, metadata, llm_pipeline, model, top_k=2):
    error_message = validate_initialization(faiss_index, metadata, model, llm_pipeline)
    if error_message:
        return error_message

    contexts = retrieve_contexts(query_text, faiss_index, metadata, model, top_k)
    return generate_response(query_text, contexts, llm_pipeline)

def ask_and_respond():
    """Ask a question, get the response"""
    num_questions = 2
    for i in range(num_questions):
        query_text = input(f"Enter question {i+1}: ").strip()

        if faiss_index is not None and metadata_store.size > 0 and model_instance:
            response = query_and_respond(
                prompt="", query_text=query_text, faiss_index=faiss_index,
                metadata=metadata_store, llm_pipeline=llm_pipeline, model=model_instance, top_k=2
            )
            print(f"Response to Question {i+1}: {response}")
        else:
            print("Error: Ensure FAISS index, metadata store, and model instance are initialized.")
            
ask_and_respond()
