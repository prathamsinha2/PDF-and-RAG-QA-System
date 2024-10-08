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

# Function to search the document using the query
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

def segment_gen(response):
    Prompt = f"From which segment of the document did you find this response: {response}"
    segm = model.generate_content(Prompt).text
    return segm

extracted_text = ""
collection_name = "text_file11111"

def upload_pdf(pdf_file):
    global extracted_text
    extracted_text = extraction(pdf_file)
    chromaup(extracted_text, collection_name)
    return "File uploaded and processed! You can now ask questions about the document."
def ask_question(query):
    if not extracted_text:
        return "Please upload a PDF first!", None, None

    resp = search_doc(query, collection_name)
    doc_segment = resp['documents'][0][0]  
    answer = generation(query, doc_segment)
    segment_info = segment_gen(answer)
    return answer, segment_info
with gr.Blocks() as rag_interface:
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", type="filepath")
        upload_button = gr.Button("Process PDF")
        status = gr.Textbox(label="Status", value="Please upload a PDF to start.", interactive=False)

    with gr.Row():
        chat_input = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
        send_button = gr.Button("Send")

    with gr.Row():
        generated_answer_output = gr.Textbox(label="Generated Answer", value="", interactive=False, lines=10, max_lines=20)
        segment_generation_output = gr.Textbox(label="Segment Information", value="", interactive=False, lines=10, max_lines=20)
    upload_button.click(upload_pdf, inputs=pdf_upload, outputs=status)
    send_button.click(ask_question, inputs=chat_input, outputs=[generated_answer_output, segment_generation_output])
rag_interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
