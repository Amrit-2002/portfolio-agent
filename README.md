# A RAG application using LangChain
## RAG Project Description
This project shows a LangChain retrieval pipeline served with FastAPI + Strawberry GraphQL with vector search in Pinecone and displayed via a Flask frontend application.
<br>    
Please find it hosted here: 
<br>
### A Screenshot of the Application
![A screenshot of the application](images/main_app.png)
A screenshot of the application is shown above. The user enters a question in the text area and clicks the "Ask" button. The question is sent to the FastAPI backend which calls the function to start executing the LangChain retrieval chain. The answer is returned through the same pipeline and displayed on the Flask Frontend application.     
<br>
The RAG application uses my personal website https://amrit-2002.github.io/Portfolio/ as the source of information to answer questions. The website is crawled and the contents are stored in Pinecone for later retrieval.
<br>

### The Indexing Process
![Abstract tech background](images/indexing.png)
A crawler crawls my website to get all loadable links. Then the contents of the links are loaded and chunked for easier processing by the models. Embedding are created from the chunks and stored in Pinecone for later retrieval.

### Try asking
1. “What projects are on his website?”    
1. “What tools does he use for data engineering?”    

### The Question-and-Answer Process
![Abstract tech background](images/retrieval.png)
The question is entered into the Flask frontend application. It's sent to the FastAPI GraphQL backend which calls the function to start executing the LangChain retrieval chain. An embedding is created from the question and a similarity search done in Pinecone to find semantically related text to the question.      
<br>
This context is added to the question to create a prompt for Google's Gemini Flash chat model to generate the answers. The answer is returned through the same pipeline and displayed on the Flask Frontend application.     
<br>
This RAG application can be used for any website. Just change the website in the code on GitHub.

### To run the application
1. Clone the repository
1. Install the required packages using `pip install -r requirements.txt`
1. Set the required environment variables in a `.env` file:
   - GOOGLE_API_KEY=Your_Google_API_key
   - PINECONE_API_KEY=Your_Pinecone_API_key
   - LANGSMITH_API_KEY=Your_Pinecone_environment
   - LANGSMITH_TRACING=true
1. Run the FastAPI backend using `uvicorn app.backend:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 200`
1. Run the Flask frontend using `python web/frontend_app.py` for development or `gunicorn -w 4 -b 127.0.0.1:5070 "web.frontend_app:app" --timeout 300` for production.
1. Go to `http://localhost:5070` to access the application

### Why LangChain?
LangChain provides built-in retrieval chain helpers (`create_history_aware_retriever`, `create_retrieval_chain`, `create_stuff_documents_chain`) that handle the standard RAG pattern out of the box, including reformulating follow-up questions into standalone queries using chat history. It also integrates with LangSmith, which traces all activities on the application and gives insight on what is going on. See the screenshot below of the tracing of the application on LangSmith — note that detailed traces require a paid LangSmith account.
![LangSmith tracing of the application](images/langsmith.png)   

We can see the steps of the application and how long each step took. This is useful for debugging and optimizing the application. We can also see the inputs and outputs of each step, which is useful for understanding how the application works.
![LangSmith tracing of the application](images/langsmith_trace.png)   

For more complex agentic applications where explicit state management and branching control flow matter, LangGraph would be a natural next step on top of LangChain.     

### Note
You'll need the necessary API keys and access to the services used in this project in your environment to run it.
I recommend using a virtual environment for better dependency management. The process is:
```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Technologies Used
- LangChain
- LangSmith
- FastAPI
- Strawberry GraphQL
- Flask
- Pinecone
- Google Gemini Flash 3.5 model
- Python
- HTML/CSS/JavaScript
- Google's gemini-embedding-2-preview model
- BeautifulSoup
- Python-dotenv