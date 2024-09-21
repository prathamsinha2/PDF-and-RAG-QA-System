# RAG PDF Question Answering System with Gradio Interface

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about PDF documents. It uses Google's Generative AI, ChromaDB for vector storage, and Gradio for a user-friendly interface. The system can be run on Google Colab for easy setup and execution, or deployed locally or on a server.

## Features

- Upload and process PDF documents
- Extract text from PDFs
- Embed text using Google's text embedding model
- Store embeddings in ChromaDB
- Answer questions about the uploaded document using Google's Gemini model
- User-friendly interface with Gradio
- Easy to run on Google Colab or deploy locally/on a server

## Prerequisites

- Python 3.7+
- Google Cloud API key for Generative AI
- Docker (for containerized deployment)

## Installation (Local/Server Deployment)

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/rag-pdf-qa.git
   cd rag-pdf-qa
   ```

2. Install the required packages:
   ```
   pip install google-generativeai chromadb pdfplumber gradio
   ```

3. Set up your Google API key:
   - Create a `.env` file in the project root
   - Add your API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage (Local/Server Deployment)

1. Run the application:
   ```
   python rag.py
   ```

2. Open the Gradio interface in your web browser (the URL will be displayed in the console).

3. Upload a PDF file and click "Process PDF".

4. Once processed, you can start asking questions about the document in the chat interface.

## Running on Google Colab

1. Open the `File_RAG_System.ipynb` notebook in Google Colab.

2. Set up your Google API key as described in the "Installation" section.

3. Run the cells in order:
   - The first cell installs the required packages.
   - The second cell imports the necessary libraries.
   - The third cell sets up the functions and configurations.
   - The fourth cell prompts you to upload a PDF file and processes it.
   - The fifth cell allows you to input questions about the uploaded PDF.

4. To ask questions about the PDF, run the fifth cell and enter your question when prompted.

## Gradio Interface

The Gradio interface provides a user-friendly way to interact with the RAG system. Here's a more detailed explanation of its components and functionality:

1. **PDF Upload**: 
   - The interface includes a file upload component specifically for PDF files.
   - Users can drag and drop or click to select a PDF file from their local machine.

2. **Process PDF Button**: 
   - After uploading a PDF, users click this button to extract text, create embeddings, and store them in ChromaDB.
   - This step is crucial before asking questions about the document.

3. **Status Display**: 
   - A text area that shows the current status of the system, such as "PDF processed successfully" or error messages.

4. **Question Input**: 
   - A text input field where users can type their questions about the uploaded PDF.

5. **Send Button**: 
   - Clicking this button submits the question for processing.

6. **Conversation Display**: 
   - A text area that shows the ongoing conversation, including user questions and system responses.
   - This creates a chat-like experience for users.

The Gradio interface makes the RAG system accessible to users who may not be comfortable with command-line interfaces or programming. It provides a seamless experience from uploading a document to asking questions and receiving answers.

## Docker Deployment

To deploy this application using Docker, follow these steps:

1. Build the Docker image:
   ```
   docker build -t ragsys .
   ```

2. Run the Docker container:
   ```
   docker run -p 7860:7860 -e ragsys

   ```

3. Access the application by opening a web browser and navigating to `http://localhost:7860`.

## Note

This application uses Google's Generative AI models. Make sure you comply with Google's usage policies and terms of service.

## License

[MIT License](https://opensource.org/licenses/MIT)
