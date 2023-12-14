## Executive Summary

Tax BASE leverages AI to innovate tax advisory services, aggregating BDO's extensive tax documentation. It features semantic embeddings and a Vector database, ChromaDB, for scalable data retrieval. Utilizing OpenAI's ChatGPT API and LangChain software, the platform processes queries with precision. The front-end web application enhances user interaction, providing a foundation for future AI advancements in language processing.

## Architecture
![](https://github.com/Richie-Lee/LLM_practice/blob/main/Project/TAX%20BASE/images/Tax_BASE_code_architecture.jpg)

## Module Descriptions

These modules collectively provide a streamlined workflow from data collection to user engagement, showcasing the system's capacity for handling large-scale data with advanced AI interaction and a user-friendly interface.

### Part I: Data Collection Detail

- **`bdo_uk_scrape.py`:** This script automates the collection of tax-related articles from BDO UK's website. It uses Selenium with Chrome WebDriver to simulate user interaction, navigating the site, triggering dynamic content loading, and clicking 'Show More' to access and scrape additional articles. BeautifulSoup parses the HTML content to extract article details, which are then prepared for export.

- **`merge_scraped_files.py`:** This utility script consolidates scraped data from multiple BDO regional sources. It ensures that all text is encoded uniformly and assembles the individual scraped files into a single comprehensive dataset. Each entry is re-indexed to create a globally unique identifier that amalgamates information from all the different BDO regional insights. 

Together, these scripts form the data collection mechanism of Tax BASE, providing the raw input for subsequent processing and analysis.

### Part II & III: Data Transformation and AI Interaction Detail

- **Data Transformation (`VectorDatabaseManager`):** This class in `TAX_BASE.py` manages the lifecycle of the vector database. It imports documents, supports text and PDF formats, splits them for processing, and creates a vector database using embeddings. These embeddings are generated via OpenAI's `text-embedding-ada-002` model, ensuring data is in an AI-friendly format for querying.

- **AI Interaction (`LLMRunner`):** This class abstracts complexities of the ChatGPT API, allowing developers to concentrate on prompt engineering. It retrieves documents from the vector database using similarity search, crafts prompts to assess relevance, and captures the essence of queries for the AI to process. The LLM, specifically `gpt-3.5-turbo`, is utilized to generate responses that are contextually relevant to the user's query.

These components transform raw data into an optimized format for retrieval and utilize advanced AI models to interpret user queries and provide relevant responses.

### Part IV: Front-End Integration Detail

The `app.py` file establishes the front-end web application of Tax BASE, using Flask to serve a responsive user interface. It integrates with the back-end through API endpoints to manage user interactions:

- **Flask Setup:** Flask is configured to handle web requests and serve the main page using `render_template`.
- **Environment Variables:** Securely stored Azure API keys are loaded to interact with the ChatGPT API for backend processing.
- **User Query Handling:** The `/get_response` endpoint processes user queries, leveraging `LLMRunner` to access the vector database and generate responses.
- **Front-End Design:** Accompanying HTML and CSS files create a user-friendly interface, enabling easy query submission and presenting AI-generated responses in a conversational format.

This module provides the crucial link between Tax BASE's robust AI backend and the end-user, enabling seamless access to the system's capabilities through a web interface.
