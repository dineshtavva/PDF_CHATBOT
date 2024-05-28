import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_core.messages import HumanMessage

@st.cache_resource
def llm_init():
    os.environ["OPENAI_API_KEY"] = 'xxxxxxxxx'

    # Initialize the LLM
    llm = OpenAI( openai_organization="xxxxxxxx")

    return llm

@st.cache_resource
def embed_doc(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        pdf_loader = PyPDFLoader(uploaded_file.name)
        documents.extend(pdf_loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    all_splits = text_splitter.split_documents(documents)
    # Create a Chroma vector store
    vector_store = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    return vector_store

def question_answer(question,vectorstore):
        
    # k is the number of chunks to retrieve
    retriever = vectorstore.as_retriever(k=4)

    docs = retriever.invoke(question)

    SYSTEM_TEMPLATE = """
    Answer the user's questions in detail based on the below context. 
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
    answer = document_chain.invoke(
        {
            "context": docs,
            "messages": [
                HumanMessage(content=question)
            ],
        }
    )
    

    return docs, answer



# Streamlit UI
st.title("RAG Application")

llm = llm_init()

# File uploader
uploaded_files = st.file_uploader("upload your files here", type=["pdf"], accept_multiple_files=True)

if uploaded_files:

    vector_store = embed_doc(uploaded_files)

    # Get user query
    user_query = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        # Retrieve and generate answer
        docs,answer = question_answer(user_query,vector_store)
        # answer = rag_chain({"question": user_query})
        st.write("returned info:", docs) 
        st.write("Answer:", answer) 
