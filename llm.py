import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from the .env file
load_dotenv()

# Fetch the API key from the environment variables
api_key = os.getenv("GROQ_API_KEY")
model="llama3-70b-8192"


llm = ChatGroq(
    model=model,
    api_key=api_key,
    temperature=0.5,  
)