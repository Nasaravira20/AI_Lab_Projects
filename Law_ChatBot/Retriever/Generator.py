from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline

load_dotenv()

HUGGINGFACE = os.getenv('HUGGINGFACE_API')
persistent_db = os.getenv('PERSIST_DIR')
client = InferenceClient(api_key=HUGGINGFACE)

def create_prompt(query):
    prompt_template = (
        "You are a legal expert providing advice to citizens facing difficult legal situations. "
        "The user will describe their problem, and your task is to offer a step-by-step guide on the legal actions "
        "they should take to resolve it. Use simple language and make sure to provide clear, actionable advice. "
        "If specific legal terminology or processes are involved, explain them briefly. Assume the user has no prior legal knowledge.\n\n"
        "Explain in detail with next steps"
        "User's Query: {query}"
    )
    return prompt_template.format(query=query)


def process(query):

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persistent_db,embedding_function=embedding_function)
    retriever = vector_store.as_retriever()

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    prompt = create_prompt(query=query)

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=280,
        temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACE,
    )

    

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever = retriever,
        chain_type = 'stuff'
    )

    result = qa_chain.invoke(prompt)
    return result['result']

if __name__ == '__main__':
    answer = process('laws about cyber-threads')
    print(answer['result'])