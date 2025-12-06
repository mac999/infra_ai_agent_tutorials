import os
import requests
import json
from pydantic.v1 import BaseModel, Field, validator
from langchain_core.tools import StructuredTool, ToolException
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Load environment variables from a .env file
load_dotenv()

key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4", openai_api_key=key)

# Create agent using LangGraph
agent = create_react_agent(llm, tools=[])

query = "What is the weather today?"
response = agent.invoke({"messages": [("user", query)]})
print(response)

ai_message = response['messages'][-1].text
print("AI Response:", ai_message)

