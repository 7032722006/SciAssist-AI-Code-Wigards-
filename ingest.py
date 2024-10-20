import logging
import json
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader


def load_config(config_path='config.json'):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_config()
print(config["storage"])
model_name = config['model_settings']['embedding_model']
model_kwargs = config['model_settings']['model_kwargs']
encode_kwargs = config['model_settings']['encode_kwargs']
src_dir = config['document_loaders']['source_directory']

Embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

DOCUMENT_MAP = {
    
    ".txt": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,

    
}

def load_documents(source_dir: str):
    documents = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in DOCUMENT_MAP:
                path = os.path.join(root, file)
                try:
                    loader_class = DOCUMENT_MAP[ext]
                    loader = loader_class(path)
                    documents.append(loader.load()[0])
                    logging.info(f"{path} loaded.")
                except Exception as e:
                    logging.error(f"Error loading {path}: {e}")
    return documents

def main():
    logging.basicConfig(level=logging.INFO)
    documents = load_documents(src_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    if config['storage'] == 'Qdrant':
        db = Qdrant.from_documents(texts, Embeddings, url=config["storage_settings"]["Qdrant"]["url"],
                                        prefer_grpc=False, 
                                        collection_name=config["storage_settings"]["Qdrant"]["collection_name"],
                                        force_recreate=True)
    elif config['storage'] == 'Chroma':
        db = Chroma.from_documents(texts, Embeddings, 
                                        collection_metadata=config["storage_settings"]["Chroma"]["collection_metadata"], 
                                        persist_directory=config["storage_settings"]["Chroma"]["persist_directory"])
    else:
        db = FAISS.from_documents(texts, Embeddings)
        db.save_local(config["storage_settings"]["FAISS"]["index_path"])

    
    logging.info("Vector DB Successfully Created!")
    

if __name__ == "__main__":
    main()
