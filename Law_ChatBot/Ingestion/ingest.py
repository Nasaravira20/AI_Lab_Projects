import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

persist_directory = '/home/mw/Documents/Projects/Law/Ingestion/data'
chroma_client = chromadb.PersistentClient(path=persist_directory)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
folder_path = "/home/mw/Documents/Projects/Law/Ingestion/source"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

whole_chunked_documents = []
if(os.path.exists(persist_directory)):
    print(persist_directory)
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path=file_path)
        document = loader.load()

        chunked_documents = text_splitter.split_documents(document)
        whole_chunked_documents.extend(chunked_documents)

db = Chroma.from_documents(
    documents=whole_chunked_documents,
    embedding=embedding_function,
    persist_directory=persist_directory,
)
db.persist()



print(f"Added {len(whole_chunked_documents)} chunks to Chroma DB")
