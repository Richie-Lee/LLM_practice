# General
import os
from dotenv import load_dotenv, find_dotenv

# Import
import langchain #!pip install --upgrade langchain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

# Splitting
from langchain.text_splitter import CharacterTextSplitter

# Embedding
import openai
from langchain.embeddings.openai import OpenAIEmbeddings

# Vector db
from langchain.vectorstores import Chroma
import chromadb

# For tracking runtime
from datetime import datetime

# Specify directories
import_directory_pdf = ""
import_directory_txt = "C:/Users/RLee/Desktop/TAX BASE/bdo_scrape_full.txt"

# Get openai_api_key using .env file
_ = load_dotenv(find_dotenv("C:/Users/RLee/Desktop/TAX BASE/openai_api_key.env")) # .env filepath
openai.api_key = os.environ["OPENAI_API_KEY"]


def import_document(import_directory):
    # Import data in LangChain format (only txt/pdf supported)
    if import_directory[-3:] == "txt":
        loader = TextLoader(import_directory_txt, encoding = "utf-8")
    elif import_directory[-3:] == "pdf":
        loader = PyPDFLoader(import_directory_pdf)
    else:
        raise Exception("Filetypes other than txt and pdf are not supported")
    
    return loader.load()



def split_document(doc):
    # split it into chunks on "\n-\n" (scraping method ensures source data formatting)
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0, 
        separator = '\n-\n' # Splitting criteria
    )
    
    return text_splitter.split_documents(doc)



def define_embedding_function():
    # Create an Embedding function (OpenAI version)
    return OpenAIEmbeddings()
   


def create_vector_db(vector_db_directory, doc_split, embedding_function):
    # https://python.langchain.com/docs/integrations/vectorstores/chroma
    vector_db = Chroma.from_documents(
        documents = doc_split,
        embedding = embedding_function,
        persist_directory = vector_db_directory # chroma-specific keyword
    )
    
    # Save to use later
    vector_db.persist()
    
    return vector_db

def load_vector_db(vector_db_directory, embedding_function):
    return Chroma(persist_directory = vector_db_directory, embedding_function = embedding_function)



def retrieval(query, vector_db, n_subset):
    # Embed question and get the <n_subset> most similar chunk embeddings from vector store
    vector_db_matches = vector_db.similarity_search(query, n_subset)
    
    # Parse the list of document types to a single string (for convenient prompt ingestion)
    vector_db_matches_str = ""
    for x in vector_db_matches:
        vector_db_matches_str = vector_db_matches_str + "\n-\n" + x.page_content
    
    return vector_db_matches_str



def get_completion(prompt, model = "gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]



def prompt_template(query, vector_db_matches_str):    
    # Prompt engineering
    prompt = f""" 
    Given a list of document summaries, your task is to assess each document strictly for relevance to the provided query. Exclude all documents that are 'likely irrelevant'—those that are only marginally related to the query.
    
    Input:
    
    Query: <{query}>
    Document Summaries: <{vector_db_matches_str}>
    
    Procedure:
    
    1. Essence Extraction: Discern the core essence and essential points of the query.
    2. Relevance Assessment: Review the titles and descriptions of each document summary. Determine whether the document is likely to be relevant or maybe relevant to the query's key points. Disregard any document that does not appear to closely align with the query's essence.
    
    Output:
    
    First print the question, and your associated interpretation of the question:
    
    - Question: [User input question]
    - Key Objective of the Query: [Concise summary of the query's key points]
    
    Insert a line here, to show a clear break using dashes 
    
    Don't include the remaining irrelevant documents in the output.
    Then, for each document that is determined to be relevant or maybe relevant, present the following details:
    
    - Conclusion: [Relevant/Maybe Relevant]
    - Reasoning: [Justification for the relevance assessment, connecting the document's title and description to the query]
    - Document Details:
      - Index: [Document index]
      - Title: [Title of the document]
      - Date: [Publication date of the document]
      - Description: [Overview of the document's main themes and points]
      - Link: [Direct URL]
    """
    
    return prompt



def run_llm(query, vector_db_matches_str, prompt):
    start_time = datetime.now() 
    
    # Get LLM response
    response = get_completion(prompt)
        
    total_runtime = datetime.now() - start_time
    
    # Format total runtime
    hours, remainder = divmod(total_runtime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    total_runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Format return statement
    return {"Query": query, "LLM Response": response, "Runtime": total_runtime, "Vector_db_matches": vector_db_matches_str}





def main(import_directory, vector_db_directory, query):
    # Create VectorDB from source file
    if initialise_vector_db == True:
        doc = import_document(import_directory)
        doc_split = split_document(doc)
        embedding_function = define_embedding_function()
        vector_db = create_vector_db(vector_db_directory, doc_split, embedding_function)
    
    # Load existing VectorDB
    else:
        embedding_function = define_embedding_function()
        vector_db = load_vector_db(vector_db_directory, embedding_function)
    
    # Runnning LLM
    n_subset = 10
    vector_db_matches_str = retrieval(query, vector_db, n_subset)
    prompt = prompt_template(query, vector_db_matches_str)
    result = run_llm(query, vector_db_matches_str, prompt)
    
    return result



initialise_vector_db = False

query = """
As a BDO UK tax professional, I'm interested in the developments following the OECD's publication of the Model Globe Rules. 
Can you tell me which jurisdictions have adopted final legislation to implement Pillar Two and which jurisdictions have published draft legislation for the same?
"""

import_directory = "C:/Users/RLee/Desktop/TAX BASE/bdo_scrape_full.txt"
vector_db_directory = "C:/Users/RLee/Downloads/vectordb"
result = main(import_directory, vector_db_directory, query)







    
    
    
    
    
    
    
    