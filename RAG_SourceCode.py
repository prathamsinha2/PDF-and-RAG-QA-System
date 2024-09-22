{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "5boX6yf8rRw5"
   },
   "outputs": [],
   "source": [
    "!pip install -q -U google-generativeai chromadb pdfplumber"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "vQ5FDWTtsEPq"
   },
   "outputs": [],
   "source": [
    "import google.generativeai as genai\n",
    "import chromadb\n",
    "from chromadb import Documents, EmbeddingFunction, Embeddings\n",
    "import pdfplumber as plmb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "yNi1W2mztDFD"
   },
   "outputs": [],
   "source": [
    "from google.colab import userdata\n",
    "key = userdata.get('GOOGLE_API_KEY')\n",
    "genai.configure(api_key=key)\n",
    "safety_settings = [\n",
    "    {\n",
    "        \"category\": \"HARM_CATEGORY_DANGEROUS_CONTENT\",\n",
    "        \"threshold\": \"BLOCK_NONE\",\n",
    "    },\n",
    "]\n",
    "model = genai.GenerativeModel('models/gemini-1.5-flash', safety_settings=safety_settings)\n",
    "\n",
    "from google.colab import files\n",
    "def upload():\n",
    "  uploaded = files.upload()\n",
    "  if uploaded:\n",
    "    for file_name in uploaded.keys():\n",
    "      print(f\"Uploaded: {file_name}\")\n",
    "      return file_name\n",
    "  else:\n",
    "    return None\n",
    "\n",
    "def extraction(pdf_path):\n",
    "  with plmb.open(pdf_path) as pdf:\n",
    "    text = \"\"\n",
    "    for page in pdf.pages:\n",
    "      text += page.extract_text()\n",
    "  return text\n",
    "\n",
    "def embed(text):\n",
    "  embed_data = genai.embed_content(model='models/text-embedding-004', content=[text], output_dimensionality=384)\n",
    "  return embed_data['embedding'][0]\n",
    "\n",
    "def chromaup(data, name):\n",
    "  client = chromadb.Client()\n",
    "  embeddings = embed(data)\n",
    "  collection = client.get_or_create_collection(name)\n",
    "  collection.add(\n",
    "    ids=[\"segment\"],\n",
    "    documents=[data],\n",
    "    embeddings=[embeddings]\n",
    "  )\n",
    "  return collection\n",
    "\n",
    "def embed_q(query):\n",
    "  embed_data = genai.embed_content(model='models/text-embedding-004', content=[query], output_dimensionality=384)\n",
    "  return embed_data['embedding'][0]\n",
    "\n",
    "def search_doc(query_text, collection_name):\n",
    "  query_embed = embed_q(query_text)\n",
    "  client = chromadb.Client()\n",
    "  coll = client.get_or_create_collection(collection_name)\n",
    "  results = coll.query(\n",
    "    query_embeddings=[query_embed],\n",
    "    n_results=1\n",
    "  )\n",
    "  return results\n",
    "\n",
    "def generation(query, doc):\n",
    "  Prompt = f\"Document: {doc}\\n\\nQuestion: {query}\"\n",
    "  response = model.generate_content(Prompt).text\n",
    "  return response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "AxA3UCaOweL2"
   },
   "outputs": [],
   "source": [
    "file = upload()\n",
    "text = extraction(file)\n",
    "text_db = chromaup(text, 'text_file1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "221i-axZwppY"
   },
   "outputs": [],
   "source": [
    "que = input()\n",
    "resp = search_doc(que, 'text_file1')\n",
    "ans = generation(que, resp['documents'][0][0])\n",
    "print(ans)"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
