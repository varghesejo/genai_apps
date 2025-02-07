# 2. Implement a RAG system for extracting information from multiple excel 
# sheets using LLM, Langchain, word embedding, excel sheet prompt and others 
# tools if necessary. If possible display the extracted information in a table format

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate

# Function to load multiple Excel sheets into a single DataFrame
def load_excel_sheets(folder_path):
    all_dataframes = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(folder_path, file)
            xls = pd.ExcelFile(file_path)
            
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                df['source'] = f"{file} - {sheet_name}"  # Metadata for tracing
                all_dataframes.append(df)
    
    return pd.concat(all_dataframes, ignore_index=True)

# Convert dataframe to textual format
def dataframe_to_text(df):
    text_data = []
    for _, row in df.iterrows():
        text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        text_data.append(text)
    return text_data

def create_index(text_data):
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = np.array(embedding_model.encode(text_data))
    # Store embeddings using FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, "faiss_index")
    return embedding_model,index

def select_llm():
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
    return READER_LLM

# Function to retrieve similar documents
def retrieve_similar_docs(query, embedding_model, index, text_data, k=5):
    query_embedding = np.array(embedding_model.encode([query]))
    _, indices = index.search(query_embedding, k)
    return [text_data[i] for i in indices[0]]

# RAG Query Function
def query_rag_system(query, llm, embedding_model, index, text_data):
    # Retrieve relevant documents
    retrieved_docs = retrieve_similar_docs(query, embedding_model, index, text_data)
    context = "\n".join(retrieved_docs)

    # Format prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate response
    response = llm(prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

# Convert LLM response to a structured table
def display_results(response):
    table_data = [[response]]
    print(tabulate(table_data, headers=["Extracted Information"], tablefmt="grid"))


def answer_user_queries(llm, embedding_model, index, text_data):
    while True:
        query = input("Enter your question: ")
        if not query:
            print("Exiting the program...")
            break

        response = query_rag_system(query,llm, embedding_model, index, text_data)
        print(response)
        # Display response
        display_results(response)

def main():
    # Load Excel files
    #folder_path = "path/to/excel/files"
    folder_path = os.path.realpath(".")
    df = load_excel_sheets(folder_path)
    print(df.head())  # Preview data

    text_data = dataframe_to_text(df)

    embedding_model, index = create_index(text_data)

    llm = select_llm()

    answer_user_queries(llm, embedding_model, index, text_data)


if __name__ == "__main__":
    main()

# query = "What is the revenue for Q1 2024?"
