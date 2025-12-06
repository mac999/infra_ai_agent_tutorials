import os
from dotenv import load_dotenv # Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# pip install langchain gradio openai tavily-python pypdf faiss-cpu
# coding QA Expert Chatbot using langchain and gradio as web UI. use PDF RAG with faiss vector DB to save, retrieve the chunk documents from the PDF. if run this chatbot, read the PDF files from ./files folder, splite them into chunks, save them to faiss as vector database. after that, create LLM using openai and create langchain prompt template, tools with web search using Tavily. create agents with them including the previous dialog memory. this UI using gradio is simliar to ChatBot.
import os, re
import glob
import gradio as gr
from gradio import ChatMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

# 1. 설정
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
VECTOR_DB_PATH = str(SCRIPT_DIR / 'faiss_index')
FILES_DIRECTORY = str(SCRIPT_DIR / 'files')
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# OpenAI 설정 - ChatOpenAI 사용
llm_model = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# 2. PDF 파일 로드 및 벡터화
def load_and_split_pdfs(files_directory):
	pdf_files = glob.glob(os.path.join(files_directory, '*.pdf'))
	documents = []
	for file in pdf_files:
		loader = PyPDFLoader(file)
		documents.extend(loader.load())
	splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
	split_documents = splitter.split_documents(documents)
	for i, doc in enumerate(split_documents):
		print(f"Document {i}: {doc.page_content[:100]}...")  # Print the first 100 characters of each split document
	return splitter.split_documents(documents)

# 3. FAISS 벡터DB 저장
def save_to_faiss(documents):
	vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
	vectordb.save_local(VECTOR_DB_PATH)
	print(f"FAISS vector database saved to {VECTOR_DB_PATH}")
	return vectordb

# 4. RAG Retrieval QA 체인 생성
def create_retrieval_qa(vectordb):
	retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.25})
	return retriever

'''
Common agent Types:

zero-shot-react-description:
	Uses the ReAct (Reasoning + Acting) framework.
	Selects tools and generates responses based on tool descriptions.
	Best for scenarios where the agent needs to reason and act without prior context.

chat-zero-shot-react-description:
	Similar to zero-shot-react-description, but optimized for chat-based interactions.
	Useful for conversational agents.

chat-conversational-react-description:
	Designed for conversational agents with memory.
	Keeps track of the conversation history to provide context-aware responses.
	This is the agent type used in your code.

self-ask-with-search:
	Designed for agents that need to ask clarifying questions before answering.
	Often used with search tools.

react-docstore:
	Optimized for retrieving and reasoning over documents in a docstore.
	Useful for document-based question answering.

conversational-react-description:
	Similar to chat-conversational-react-description, but without explicit chat optimizations.
	Includes memory for context-aware responses.
'''

# 5. Query with tools
conversation_history = []

def query_with_tools(user_input):
	# Try PDF QA first
	try:
		docs = qa_chain.invoke(user_input)
		context = "\n".join([doc.page_content for doc in docs])
		prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {user_input}\n\nAnswer:"
		result = llm_model.invoke([HumanMessage(content=prompt)])
		return result.content
	except Exception as e:
		print(f"PDF QA error: {e}")
	
	# Fallback to web search
	try:
		search_tool = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY)
		search_results = search_tool.invoke(user_input)
		return str(search_results)
	except Exception as e:
		print(f"Web search error: {e}")
	
	# Direct LLM response
	result = llm_model.invoke([HumanMessage(content=user_input)])
	return result.content

def extract_action_input(text):
	# "action_input": "..." 패턴을 정규식으로 추출
	pattern = r'"action_input"\s*:\s*"([^"]+)"'
	match = re.search(pattern, text, re.DOTALL)
	if match:
		return match.group(1)
	return None

# 초기화 과정
if not os.path.exists(VECTOR_DB_PATH):
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    docs = load_and_split_pdfs(FILES_DIRECTORY)
    save_to_faiss(docs)

vectordb = FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
qa_chain = create_retrieval_qa(vectordb)

def chatbot_interface(user_input, history):
    if not user_input:
        return history, history
    
    try:
        response = query_with_tools(user_input)
    except Exception as e:
        msg = f"Error: {str(e)}"
        print(msg)
        response = extract_action_input(str(e))
        if response == None:
            response = str(e)
    
    history.append(ChatMessage(role="user", content=user_input))
    history.append(ChatMessage(role="assistant", content=response))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("QA Expert Chatbot (PDF + Web Search)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="질문을 입력하세요...")

    clear = gr.Button("초기화")

    state = gr.State([])
    msg.submit(chatbot_interface, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch(share=True)
