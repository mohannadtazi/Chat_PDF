
from langchain.chains import RetrievalQA

import streamlit as st
import tempfile
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import llm 



#-----------------------------------------------------

@st.cache_resource
def load_pdf(url):
    index = None
    if url is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(uploaded_file, temp_file)
            temp_file_path = temp_file.name
            loaders = [PyPDFLoader(temp_file_path)]

            index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
            
    return index

#-----------------------------------------------------



st.title("I am Your Tutor, Don't Hesitate to Ask Me Anything!")

if 'message' not in st.session_state:
    st.session_state.message = []

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

for message in st.session_state.message:
    if message['role'] == 'user':
        st.chat_message('You').markdown(f'{message["message"]}')
    else:
        st.chat_message('Me').markdown(f'{message["message"]}')

prompt = st.chat_input("What do you want to know?")

if not uploaded_file:
    st.warning("Please upload a PDF file.")
    st.stop()
index = load_pdf(uploaded_file)
chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question')


if prompt:
   st.chat_message('You').markdown(f'{prompt}')
   st.session_state.message.append({'role':'user', 'message' : prompt})
   response = chain.run(prompt)
   st.chat_message('Me').markdown(f'{response}')
   st.session_state.message.append({'role':'assistant', 'message' : response})

