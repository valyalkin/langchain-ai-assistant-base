# AI assistant

This repo contains basic code for setting up ai assistants using langchain framework. The code can fully run locally.
Sqlite is used as a database, chroma db as a vector store and FastAPI as a web framework.

## Prerequisites

Please make sure that latest python is installed on your machine and OPENAI api key is set up as an environmnetal variable (OPENAI_API_KEY).

## Running the application

### Backend

```bash
# Navigate to backend repository
cd langchain-backend

# Create a virtual environment
python3 -m venv .venv

# Initialize virtual environment and install dependencies
source .venv/bin/activate
pip install -r requirements.txt

# Run the application
fastapi dev main.py
```

### Frontend
```bash
# Navigate to frontend repository
cd streamlit-frontend

# Create a virtual environment
python3 -m venv .venv

# Initialize virtual environment and install dependencies
source .venv/bin/activate
pip install -r requirements.txt

# Run the application, on the successul run, browser window will open with chat window
streamlit run main.py
```

 Happy chatting and have fun!


