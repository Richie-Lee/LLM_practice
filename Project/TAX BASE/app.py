"""Module to run the web application."""
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
)

# For credentials 
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
env_file = find_dotenv("azure_api_keys.env")
load_dotenv(env_file)

# ChatBot Logic
from TAX_BASE import LLMRunner, VectorDatabaseManager  

app = Flask(__name__)

# Fetch Azure API credentials from environment variables
azure_api_base = os.getenv("AZURE_OPENAI_API_BASE")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize the VectorDatabaseManager with Azure API key
import_directory = "C:/Users/RLee/Desktop/TAX BASE/bdo_scrape_full.txt"
vector_db_directory = "C:/Users/RLee/Downloads/vectordb"
vector_db_manager = VectorDatabaseManager(import_directory, vector_db_directory, azure_api_key)
embedding_function = vector_db_manager.define_embedding_function()
vector_db = vector_db_manager.load_vector_db(embedding_function)

# Initialize LLM with Azure API base and key
llm_runner = LLMRunner(vector_db, azure_api_base, azure_api_key)

@app.route("/")
def home() -> str:
    """ Render the main page. """
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response() -> Response:
    """ Handle POST requests to get a chatbot response. """
    user_message = request.form["user_message"]
    try:
        n_subset = 10  # Define how many documents to consider from the vector database
        response = llm_runner.run_llm(user_message, n_subset)
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    return jsonify({"response": str(response)})

if __name__ == "__main__":
    app.run(debug=True)
