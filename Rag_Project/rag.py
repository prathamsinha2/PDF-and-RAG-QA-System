import google.generativeai as genai
import chromadb
import pdfplumber as plmb
import gradio as gr
key="YOUR_API_KEY_HERE"
genai.configure(api_key=key)
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
model = genai.GenerativeModel('models/gemini-1.5-flash', safety_settings=safety_settings)

def extraction(pdf_file):
    with plmb.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text
    
def embed(text):
    embed_data = genai.embed_content(model='models/text-embedding-004', content=[text], output_dimensionality=384)
    return embed_data['embedding'][0]

def chromaup(data, name):
    client = chromadb.Client()
    embeddings = embed(data)
    collection = client.get_or_create_collection(name)
    collection.add(
        ids=["segment"],
        documents=[data],
        embeddings=[embeddings]
    )
    return collection
    
def embed_q(query):
    embed_data = genai.embed_content(model='models/text-embedding-004', content=[query], output_dimensionality=384)
    return embed_data['embedding'][0]

def search_doc(query_text, collection_name):
    query_embed = embed_q(query_text)
    client = chromadb.Client()
    coll = client.get_or_create_collection(collection_name)
    results = coll.query(
        query_embeddings=[query_embed],
        n_results=1
    )
    return results

def generation(query, doc):
    Prompt = f"Document: {doc}\n\nQuestion: {query}"
    response = model.generate_content(Prompt).text
    return response

extracted_text = ""
collection_name = "text_file11111"
chat_history = []  

def upload_pdf(pdf_file):
    global extracted_text
    extracted_text = extraction(pdf_file)
    chromaup(extracted_text, collection_name)
    return "File uploaded and processed! You can now ask questions about the document."

def ask_question(query):
    global chat_history
    if not extracted_text:
        return "Please upload a PDF first!"
        
    resp = search_doc(query, collection_name)
    answer = generation(query, resp['documents'][0][0])
    chat_history.append(f"User: {query}")
    chat_history.append(f"Bot: {answer}")
    return "\n".join(chat_history)


with gr.Blocks() as rag_interface:
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", type="filepath")
        upload_button = gr.Button("Process PDF")
        status = gr.Textbox(label="Status", value="Please upload a PDF to start.", interactive=False)
    with gr.Row():
        chat_history_output = gr.Textbox(label="Conversation", value="", interactive=False, lines=10, max_lines=20)
        chat_input = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
        send_button = gr.Button("Send")
    upload_button.click(upload_pdf, inputs=pdf_upload, outputs=status)

   
    send_button.click(ask_question, inputs=chat_input, outputs=chat_history_output)
rag_interface.launch(share=True)
