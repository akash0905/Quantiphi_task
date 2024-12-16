# Quantiphi_task
RAG-LLM Model and pipeline : These code with extract text from pdf and the text presents in the images of pdf so you can ak questions and get answer from complete pdf.

In **Extract_store_embedding.py** File below Operations are getting perform.

1. Extract Text from PDF:

    Opens a PDF file using PyMuPDF (fitz library).
    Extracts text from each page of the PDF.
    Extracts and processes embedded images in the PDF using Pillow and pytesseract to perform OCR (Optical Character Recognition).

2. Store Metadata:

    Captures metadata for each page and each extracted image, including page numbers and content sources (e.g., "PDF", "Image X").

3. Split Text into Chunks:
    Uses the RecursiveCharacterTextSplitter to split large text data into smaller chunks for efficient processing.
    Includes overlap between chunks to ensure context is preserved across splits.

4. Generate Text Embeddings:
    Encodes text chunks into numerical vectors (embeddings) using the SentenceTransformer model (all-MiniLM-L6-v2).

5. Store Embeddings in FAISS:
    Creates a FAISS index (a library for fast similarity search) for storing and retrieving the text embeddings.
    Saves the FAISS index and metadata to disk (faiss_index.bin and metadata_store.npy).

This code effectively extracts, processes, and indexes text and image data from PDFs and store it locally using FAISS.  
--------------------------------------------------------------------------------------------------------------------------

In **llM_model_Query.npy** File below Operations are getting perform.


1. Load FAISS Index and Metadata:
    Tries to load a previously saved FAISS index (faiss_index.bin) and metadata (metadata_store.npy).
    If successful, prints a confirmation message; otherwise, handles errors gracefully.

2. Load SentenceTransformer Model:
    Loads the SentenceTransformer model (all-MiniLM-L6-v2) for generating embeddings.
    Moves the model to GPU if available, otherwise defaults to the CPU.

3. Load Large Language Model (LLM) Pipeline:
    Initializes a transformer-based LLM pipeline (tiiuae/falcon-7b-instruct) using the Hugging Face pipeline for generating responses.
    

4. Retrieve Relevant Contexts:
    Function retrieve_contexts retrieves the most relevant text chunks for a given query using:
    Sentence embedding of the query.
    FAISS index to perform similarity search, returning the top k contexts.
    Uses metadata to extract the original text associated with the indices retrieved from the FAISS index.

5. Generate Response with LLM:
    Function generate_response combines the retrieved contexts into a single prompt.
    Passes the prompt to the LLM pipeline to generate a response.
    
6. Full Query-Response Workflow:
    query_and_respond orchestrates the full workflow:
    Validates that all components are initialized.
    Retrieves relevant contexts from the FAISS index.
    Generates a response based on the contexts and the query.
    
7. Interactive Q&A Session:
    ask_and_respond prompts the user to enter questions interactively:
    Asks the user for two questions (can be adjusted).
    Runs the full query-and-response workflow for each question.
    Outputs the LLM-generated response or appropriate error messages.
    High-Level Purpose:
    This code creates an end-to-end system for answering user queries by:

    Retrieving relevant information from a knowledge base (FAISS index).
    Using a large language model to generate detailed, context-aware responses.
    It's designed to handle tasks like semantic search, knowledge-based question answering, and interactive assistance.

--------------------------------------------------------------------------------------------------------------------------
**Run the code**
1. First Create virtual_environment.
2. Install Dependnecies using requirements.txt
3. Run the First code : Extract_store_embedding.py
4. Run the Second Code : llM_model_Query.npy
5. you can modify the code to ask multiple Questions currently limit is set to 2.

**Note:**
1. I have used tiiuae/falcon-7b-instruct Model because it's accuracy is good as compared to other model.
2. As model size is around 16+ GB so it's taking bit time in returing the response around 60-90 seconds. 
3. i tried running the code on Google colab but it was not supporting PyTesreact because it's path has to be pass in environment path so because of time constrains i have run the code on Local PC
4. Response Screenshot are attached as Image