from flask import Flask, request, jsonify, render_template
import os
import json
from qdrant_client import QdrantClient
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.vectorstores import Chroma

app = Flask(__name__)
def load_config(config_path='config.json'):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_config()
print(config["storage"])
# Initialize LLM and other components as in the original code


confi = {}
llm = CTransformers(
    model=config["llm_settings"]["local_llm"],
    model_type=config["llm_settings"]["model_type"],
    lib=config["llm_settings"]["lib"],
    **confi
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
you are able to answer the question in best possible way based on the context only.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""



model_name = config['model_settings']['embedding_model']
model_kwargs = config['model_settings']['model_kwargs']
encode_kwargs = config['model_settings']['encode_kwargs']
src_dir = config['document_loaders']['source_directory']

Embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])



def load_vector_storage(config, Embeddings):
    """
    Load the appropriate vector store based on the configuration.

    Parameters:
    - config: A dictionary containing the configuration settings.
    - Embeddings: The embeddings or a function to generate embeddings.

    Returns:
    - An instance of the vector store based on the configuration.
    """
    if config['storage'] == 'FAISS':
        # Assuming FAISS is a module or class with a load_local method
        index_path = config["storage_settings"]["FAISS"]["index_path"]
        load_vector_store = FAISS.load_local(index_path, Embeddings)
    
    elif config['storage'] == 'Chroma':
        # Assuming Chroma is a class initialized with a directory and an embedding function
        persist_directory = config["storage_settings"]["Chroma"]["persist_directory"]
        load_vector_store = Chroma(persist_directory=persist_directory, embedding_function=Embeddings)
    
    elif config['storage'] == 'Qdrant':
        # Assuming QdrantClient and Qdrant are classes for interacting with Qdrant storage
        client = QdrantClient(url=config["storage_settings"]["Qdrant"]["url"], prefer_grpc=False)
        collection_name = config["storage_settings"]["Qdrant"]["collection_name"]
        load_vector_store = Qdrant(client=client, embeddings=Embeddings, collection_name=collection_name)
    
    else:
        raise ValueError("Unsupported storage type specified in configuration.")
    
    return load_vector_store

vector_store = load_vector_storage(config['storage'], Embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":1})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form.get('query')
    # Your logic to handle the query
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
