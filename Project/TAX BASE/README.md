## Executive Summary

Tax BASE leverages AI to innovate tax advisory services, aggregating BDO's extensive tax documentation. It features semantic embeddings and a Vector database, ChromaDB, for scalable data retrieval. Utilizing OpenAI's ChatGPT API and LangChain software, the platform processes queries with precision. The front-end web application enhances user interaction, providing a foundation for future AI advancements in language processing.

## Architecture
![](https://github.com/Richie-Lee/LLM_practice/blob/main/Project/TAX%20BASE/images/Tax_BASE_code_architecture.jpg)

## Module Descriptions

### Part I: Data Collection
- `bdo_XX_scrape.py`: Uses Selenium and BeautifulSoup for dynamic web scraping of BDO's tax content.
- `merge_scraped_files.py`: Consolidates data from various BDO sources into a single, uniformly encoded dataset.

### Part II & III: Data Transformation and AI Interaction
- `VectorDatabaseManager`: Manages document import and embedding using OpenAI models to create a queryable vector database.
- `LLMRunner`: Interfaces with ChatGPT's API, facilitating prompt engineering and relevant document retrieval.

### Part IV: Front-End Integration
- `app.py`: A Flask application that acts as the front-end, integrating environment variables and handling user queries.
- HTML/CSS: Shapes the user interface for query input and displays AI-generated responses.

These modules collectively provide a streamlined workflow from data collection to user engagement, showcasing the system's capacity for handling large-scale data with advanced AI interaction and a user-friendly interface.
