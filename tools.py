import os
from langchain.utilities import PythonREPL
from langchain.agents import Tool, create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.document_loaders import UnstructuredFileLoader
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from custom import Llama2API

load_dotenv(find_dotenv())
local = CTransformers(
    model=os.getenv('LOCAL_MODEL'), 
    model_type=os.getenv('LOCAL_MODEL_TYPE'), 
    config={'context_length': 2048, 'max_new_tokens': 4096, 'temperature': 0.05}
)

hfembeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

gcc = Llama2API(
    temperature=0.7,
    max_output_tokens=1024,
    base_url=os.getenv('LLAMA2API_API_BASE'),
    api_key=os.getenv('LLAMA2API_API_KEY'),
)

# Ask a document
def ask_document(str):
    doc, query = str.split(",")
    loader = UnstructuredFileLoader(doc)
    documents = loader.load()    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents, hfembeddings)
    qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""
    prompt = PromptTemplate(template=qa_template, input_variables=['context', 'question'])
    qa = RetrievalQA.from_chain_type(
        llm=gcc, 
        chain_type="map_reduce", 
        retriever=vectorstore.as_retriever(search_kwargs={'k':2}),
        chain_type_kwargs={'prompt': prompt})
    return qa.run(query)

ask_document_tool = Tool(
    name="ask_document",
    description="""Asks queries about a given document. Can take in multiple document 
types including PDF, text, Word, Excel, Powerpoint, images and so on except CSV files. 
The input to this tool should be a comma separated list of length two. The first string 
in the list is the file path for the document you want  you want query and the second 
is the query itself. For example, `dir/attention.pdf,What is the summary of 
the document?` would be the input if you wanted to query the dir/attention.pdf file.""",
    func=ask_document
)

# Ask a CSV file
def ask_csv(str):
    doc, query = str.split(",")
    agent = create_csv_agent(
        local,
        doc,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent.run(query)

ask_csv_tool = Tool(
    name="ask_csv",
    description="""Asks queries about a given csv file. The input to this tool 
should be a comma separated list of length two. The first string in the list is 
the file path for the csv file you want  you want query and the second is the 
query itself. For example, `dir/data.csv,How many rows are there in the csv?` 
would be the input if you wanted to query the dir/data.csv file. When calling
functions to the csv data, use the python_repl tool.""",
    func=ask_csv
)   

# Python tool
python_tool = Tool(
    name="python_repl",
    description="""A Python shell. Use this to execute python commands only 
when explicitly asked to. Input should be a valid python command. If you want 
to see the output of a value, you should print it ut with `print(...)`.""",
    func=PythonREPL().run)

# DuckDuckGo search
search_tool = DuckDuckGoSearchRun()

# get tools for the agent
def get_tools():
    return [
        ask_document_tool,
        ask_csv_tool,
        python_tool,
        search_tool
    ]
