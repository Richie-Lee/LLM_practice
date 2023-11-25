import os
# from dotenv import load_dotenv, find_dotenv
# import langchain
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# import chromadb
from datetime import datetime

class VectorDatabaseManager:
    """
    Manages the vector database's lifecycle including creation, loading, and document processing.
    """
    def __init__(self, import_directory, vector_db_directory, api_key):
        self.import_directory = import_directory
        self.vector_db_directory = vector_db_directory
        self.api_key = api_key

    def import_document(self):
        """
        Imports a document from the specified directory, supports txt and pdf formats.
        """
        if self.import_directory.endswith("txt"):
            loader = TextLoader(self.import_directory, encoding="utf-8")
        elif self.import_directory.endswith("pdf"):
            loader = PyPDFLoader(self.import_directory)
        else:
            raise Exception("Filetypes other than txt and pdf are not supported")
        return loader.load()

    def split_document(self, doc):
        """
        Splits the imported document into chunks based on a specified separator.
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separator='\n-\n'
        )
        return text_splitter.split_documents(doc)

    def define_embedding_function(self):
        """
        Defines and returns the embedding function to be used for the documents.
        Passes the Azure API key to OpenAIEmbeddings.
        """
        return OpenAIEmbeddings(openai_api_key=self.api_key)

    def create_vector_db(self, doc_split, embedding_function):
        """
        Creates and persists a vector database from the split documents using the given embedding function.
        """
        vector_db = Chroma.from_documents(
            documents=doc_split,
            embedding=embedding_function,
            persist_directory=self.vector_db_directory
        )
        vector_db.persist()
        return vector_db

    def load_vector_db(self, embedding_function):
        """
        Loads an existing vector database using the given embedding function.
        """
        return Chroma(persist_directory=self.vector_db_directory, embedding_function=embedding_function)



class LLMRunner:
    """
    Handles the retrieval from the vector database and operations with the Large Language Model (LLM).
    """
    def __init__(self, vector_db, azure_api_base, azure_api_key):
        self.vector_db = vector_db
        os.environ["OPENAI_API_BASE"] = azure_api_base  # Replace with your URL
        openai.api_key = azure_api_key  # Replace with one of your keys

    @staticmethod
    def get_completion(prompt, model="gpt-3.5-turbo"):
        """
        Generates a completion for the given prompt using the specified model.
        """
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message["content"]

    def retrieval(self, query, n_subset):
        """
        Retrieves the most similar documents from the vector database for a given query.
        """
        vector_db_matches = self.vector_db.similarity_search(query, n_subset)
        vector_db_matches_str = "\n-\n".join(x.page_content for x in vector_db_matches)
        return vector_db_matches_str

    @staticmethod
    def prompt_template(query, vector_db_matches_str):
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

    def run_llm(self, query, n_subset):
        """
        Runs the LLM process including retrieval from the vector database, prompt generation, and response retrieval.
        """
        vector_db_matches_str = self.retrieval(query, n_subset)
        prompt = LLMRunner.prompt_template(query, vector_db_matches_str)

        start_time = datetime.now()
        response = LLMRunner.get_completion(prompt)
        total_runtime = datetime.now() - start_time
        hours, remainder = divmod(total_runtime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return {"Query": query, "LLM Response": response, "Runtime": formatted_runtime, "Vector_db_matches": vector_db_matches_str}