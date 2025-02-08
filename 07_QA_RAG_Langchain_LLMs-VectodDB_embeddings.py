# Assignment1:  Implement a question answering system with RAG, word embedding, vector database, langchain, llm and any other tools 

# Dependencies:
# pip install -qU langchain-groq 
# pip install -qU langchain-huggingface 


# import libraries
import getpass 
import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.vectorstores import InMemoryVectorStore 
import bs4 
from langchain import hub 
from langchain_community.document_loaders import WebBaseLoader 
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langgraph.graph import START, StateGraph 
from typing_extensions import List, TypedDict 


def select_chat_model():
    if not os.environ.get("GROQ_API_KEY"): 
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ") 
    llm = ChatGroq(model="llama3-8b-8192") 
    return llm

def select_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
    return embeddings

def select_vector_store(embeddings):
    vector_store = InMemoryVectorStore(embeddings) 
    return vector_store

def load_refernce_data():
    # Load and chunk contents of the blog 
    loader = WebBaseLoader( 
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), 
        bs_kwargs=dict( 
            parse_only=bs4.SoupStrainer( 
                class_=("post-content", "post-title", "post-header") 
            ) 
        ), 
    ) 
    docs = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    all_splits = text_splitter.split_documents(docs) 
    return all_splits




def main():
    # Configure key for the API using .env file
    load_dotenv()
    
    llm = select_chat_model()

    embeddings = select_embedding_model()

    vector_store = select_vector_store(embeddings)

    all_splits = load_refernce_data()

    # Index chunks 
    _ = vector_store.add_documents(all_splits)

    # Define prompt for question-answering 
    prompt = hub.pull("rlm/rag-prompt") 

    # Define state for application 
    class State(TypedDict): 
        question: str 
        context: List[Document] 
        answer: str 

    # Define application steps 
    def retrieve(state: State): 
        retrieved_docs = vector_store.similarity_search(state["question"]) 
        return {"context": retrieved_docs} 

    def generate(state: State): 
        docs_content = "\n\n".join(doc.page_content for doc in state["context"]) 
        messages = prompt.invoke({"question": state["question"], "context": docs_content}) 
        response = llm.invoke(messages) 
        return {"answer": response.content} 
    
    # Compile application and test 
    graph_builder = StateGraph(State).add_sequence([retrieve, generate]) 
    graph_builder.add_edge(START, "retrieve") 
    graph = graph_builder.compile() 

    print("\nQuestion-Answering System is ready to use!\n\n")
    print("<<TOPIC>> : 'LLM Powered Autonomous Agents blog post' \n")
    print("Ask your questions about this topic \n\n")
    while True:
        query = input("Question:>>> ")
        if not query:
            print("\nNot a valid query...Exiting the system!")
            break
        response = graph.invoke({"question": query}) 
        print("\nResponse:>>>",response["answer"],"\n\n") 

if __name__ == "__main__":
    main()
