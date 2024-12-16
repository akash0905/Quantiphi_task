import fitz
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

def extract_text_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image, lang="eng")
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)
        text_chunks = []
        metadata = []

        for page_num in range(len(document)):
            page = document[page_num]
            page_text = page.get_text()
            text_chunks.append(page_text)
            metadata.append({"page_number": page_num + 1, "source": "PDF", "text": page_text})

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_text = extract_text_from_image(image_bytes)
                text_chunks.append(image_text)
                metadata.append({
                    "page_number": page_num + 1,
                    "source": f"Image {img_index + 1}",
                    "text": image_text
                })

        document.close()
        return text_chunks, metadata
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return [], []

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
        )
        return splitter.split_text(text)
    except Exception as e:
        print(f"Error splitting text into chunks: {e}")
        return []

def create_text_embeddings(chunks):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = [model.encode(chunk) for chunk in chunks]
        return embeddings, model
    except Exception as e:
        print(f"Error creating text embeddings: {e}")
        return [], None

def store_in_faiss(embeddings, metadata):
    try:
        embedding_matrix = np.array(embeddings)
        dimension = embedding_matrix.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embedding_matrix)

        # Save FAISS index
        faiss.write_index(faiss_index, "faiss_index.bin")
        np.save("metadata_store.npy", metadata)

        return faiss_index, metadata
    except Exception as e:
        print(f"Error storing embeddings in FAISS: {e}")
        return None, []

# PDF path
pdf_path = r"C:\Users\user\Downloads\Quantiphi_Assingment\chapter_1_2.pdf"

extracted_chunks, metadata = extract_text_from_pdf(pdf_path)
embeddings, model_instance = create_text_embeddings(extracted_chunks)
faiss_index, metadata_store = store_in_faiss(embeddings, metadata)

if faiss_index:
    print("FAISS index and metadata stored successfully.")
