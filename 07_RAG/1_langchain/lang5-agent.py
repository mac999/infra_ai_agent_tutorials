# !pip install langchain openai wikipedia langchain-community numexpr "httpx==0.27.2"

import os
from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 

llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=2000)

# zeroshot ReAct agent with Wikipedia and LLM Math tools
from langchain.agents import initialize_agent, load_tools, AgentType
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# initialize the agent with the tools and LLM
# ZERO_SHOT_REACT_DESCRIPTION 
agent = initialize_agent(tools , llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True) 
agent.invoke("요즘 여행가기 좋은 날은 몇월이야?")

# Conversational ReAct
from langchain.memory import ConversationBufferMemory

tools = load_tools(["llm-math"],  llm=llm)                      # llm-math 도구만 사용
memory = ConversationBufferMemory(memory_key="chat_history")    # 대화 기록 메모리 키 이름 설정
conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,     # 디버깅 정보 출력
    max_iterations=3, # 에이전트 질문 해결 위한 최대 반복 횟수
    memory=memory,)
conversational_agent.invoke("Who is Geoffrey Hinton and how old is he? In addition, today is 2025-05-01.")

# Self Ask With Search Agent
from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.tools.tavily_search import TavilyAnswer

import os
os.environ["TAVILY_API_KEY"] = "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" 

tools = [TavilyAnswer(max_results=5, name="Intermediate Answer")]   # TavilyAnswer 도구 사용
prompt = hub.pull("hwchase17/self-ask-with-search")                 # "Self Ask With Search" 프롬프트를 가져옴
print(prompt.template) # 프롬프트 템플릿 출력


agent = create_self_ask_with_search_agent(llm, tools, prompt) # "Self Ask With Search" 에이전트 생성

agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, max_iterations=5, max_execution_time=10) # AgentExecutor 로 에이전트, 도구, 파싱 에러 처리 설정
agent_executor.invoke({"input": "Who lived longer, Muhammad Ali or Alan Turing?"})

# ReAct document store
from langchain.docstore.wikipedia import Wikipedia        # Wikipedia 문서 저장소를 사용
from langchain.agents import Tool
from langchain.agents.react.base import DocstoreExplorer  # 문서 저장소 탐색기 생성

docstore=DocstoreExplorer(Wikipedia())

tools = [
    Tool(
        name="Search",
        func=docstore.search,  # 키워드 기반 검색. 
        description="docstore에서 용어 검색",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup, # 정확한 용어 방식 검색.
        description="docstore에서 용어 검색",
    )
]

react = initialize_agent(tools = tools,
                        llm = llm,
                        agent = AgentType.REACT_DOCSTORE, # 에이전트 유형 ReAct document store
                        handle_parsing_errors=True,       # 구문 오류 처리
                        max_iterations=1,                 # 에이전트 최대 반복 횟수
                        max_execution_time=5,             # 에이전트 작업 수행 최대 시간(초)
                        verbose = True,)

def query_data(query):
    try:
        response = react.invoke(query)  # 질의 전달 에이전트 실행
        print(f"response={response}")  
        return response                 # 검색 결과 반환
    except Exception as e:
        print(f"Error: {e}")            
        raise 

query = "샴 고양이에게 먹이주는 법은?"
response = query_data(query)
print(response['output']) 




