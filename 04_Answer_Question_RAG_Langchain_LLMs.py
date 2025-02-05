# 12. Assignments using RAG and Langchain: Use Lang chain and Retrieval 
# Augmented Generation, (RAG), and answer a question from the user using 
# multiple PDF documents connected to the LLM using OpenAI. You can read the 
# question and refer the multiple PDF documents to answer the question. You 
# can use the APIs of any other open source LLM too.

# Assignment.12. Assignments using RAG and Langchain: Use Lang chain and 
# Retrieval Augmented Generation, (RAG), and answer a question from the user 
# using multiple PDF documents connected to the LLM using OpenAI. You can read
#  the question and refer the multiple PDF documents to answer the question. 
# You can use the APIs of any other open source LLM too.

# There is version dependency with python and other dependent libraries, 
# so using python 3.12.x based environment for installing the libraries

# Install the required libraries
#	conda install -c conda-forge langchain
#	conda install -c conda-forge langchain-community 
#	conda install -c conda-forge transformers
#	conda install -c conda-forge sentence-transformers
#	conda install -c conda-forge chromadb
#	pip install cohere
#   pip install pypdf

import os
import getpass
import warnings 
from dotenv import load_dotenv
from platform import python_version
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.llms import Cohere

def get_reference_docs():
    # Getting PDF documents
    pdf_files = input("\nEnter the PDF documents to refer, space separated filenames: ").split()
    if not pdf_files:
        print("No PDF files provided. Exiting program...")
        exit(0)

    print(f"PDF files to refer: {pdf_files}\n")
    return pdf_files

def generate_vector_db(pdf_files):
    loaders = [PyPDFLoader(path) for path in pdf_files]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    # splitt the doc into smaller chunks of size 500
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)

    # Get the huggingface embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # embed the chunks as vectors and store them in a vector database
    # Using chroma vector store
    CHROMA_PATH = "Chroma"
    db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    return db_chroma

def answer_user_queries(vector_db):
    # Create a prompt template
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Don’t justify your answers.
    Don’t give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Select the model
    model = Cohere()

    while True:
        # Get the question from the user
        query = input("\n\nQUESTION>>>")
        if not query:
            print("Exiting the program...")
            break

        # retrieve context - top 5 most relevant (closests) chunks to the query vector
        docs_chroma = vector_db.similarity_search_with_score(query, k=5)

        # generate an answer based on given user query and retrieved context information
        context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

        # load retrieved context and user query in the prompt template
        prompt = prompt_template.format(context=context_text, question=query)
        model = Cohere()
        response_text = model.predict(prompt)
        print("\n\nRESPONSE>>>\n",response_text)
    return

def main():
    warnings.filterwarnings('ignore')
    # Configure key for the API using .env file
    load_dotenv()
    print("\n\nPython version being used: ", python_version())

    pdf_files = get_reference_docs()

    vector_db = generate_vector_db(pdf_files)

    answer_user_queries(vector_db)


if __name__ == "__main__":
    main()