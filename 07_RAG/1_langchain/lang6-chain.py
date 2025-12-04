# !pip install langchain langchain_community langchain_openai
from google.colab import userdata
key = userdata.get('openai-api')


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=key, max_tokens=512)
llm.invoke("유클리드 기하학이란?")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("파이썬 코딩을 통해 설명하세요: {query}")
print(prompt)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=key, max_tokens=512)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
chain.invoke({"query": "선형대수란?"})

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableMap
from operator import itemgetter

llm = ChatOpenAI(openai_api_key=key, temperature=0.1, model="gpt-4o-mini")

basictopic = (
    ChatPromptTemplate.from_template("{topic} 에 대해 설명합니다")
    | llm
    | StrOutputParser()
    | {"base": RunnablePassthrough()}
)

positive = (
    ChatPromptTemplate.from_template(
        "{base} 의 장점은?"
    )
    | llm
    | StrOutputParser()
)

negative = (
    ChatPromptTemplate.from_template(
        "{base} 의 단점은?"
    )
    | llm
    | StrOutputParser()
)

final = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "긍정:\n{results_1}\n\n부정:\n{results_2}"),
            ("system", "비평에 대한 최종 답변 생성"),
        ]
    )
    | llm 
    | StrOutputParser()
)

# positive, negative, final을 RunnableParallel로 병렬처리
chain = (
    basictopic
    | RunnableParallel(
        results_1 = positive,
        results_2 = negative,
        original_response = itemgetter("base"),
    )
    | RunnableMap(
        {
            "positive_result": itemgetter("results_1"),
            "negative_result": itemgetter("results_2"),
            "final_answer": itemgetter("original_response"),
        }
    )
)

result = chain.invoke({"topic": "민주주의"})

positive_result = result['positive_result']
negative_result = result['negative_result']
final_answer = result['final_answer']

print("positive opinion: ", positive_result)
print("\nnegative opinion: ", negative_result)
print("\nfinal opinion", final_answer)





