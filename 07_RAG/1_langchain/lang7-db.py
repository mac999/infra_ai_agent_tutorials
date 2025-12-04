# !pip install langchain langchain_community langchain_openai pymupdf sentence-transformers
import torch
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from typing import List
from langchain_core.output_parsers import BaseOutputParser
import os

# load pdf file and split into chunks
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "files", "mama-mia.pdf")

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
text_contents = [doc.page_content for doc in texts]

# embedding model 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# vedtorstore 생성
vectorstore = FAISS.from_texts(text_contents, embeddings)

# LLM 
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name) # 토크나이저
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
llm = HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation", # 텍스트 생성
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256, # 성할 텍스트의 최대 토큰 수
        do_sample=True, # 확률 기반 샘플링 생성
        temperature=0.7,
        top_p=0.95, # 확률 분포에서 상위 95%의 누적 확률에 해당하는 토큰만 고려하여 텍스트를 생성
        device=0 if torch.cuda.is_available() else -1
    )
)

# template 정의
custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""당신은 AI 언어 모델 어시스턴트입니다. 사용자가 제공한 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색할 수 있도록 질문을 3가지 다른 버전으로 생성하는 것이 당신의 임무입니다. 사용자의 질문을 다양한 관점에서 재구성하여 거리 기반 유사도 검색의 한계를 극복할 수 있도록 돕는 것이 목표입니다. 각 버전의 질문은 줄바꿈으로 구분하여 작성하세요. 한국어로 작성하세요. 원본 질문: {question}"""
)

# OUtputParser 정의
class LineListOutputParser(BaseOutputParser):
    def parse(self, text: str) -> List[str]:
        return text.strip().split("\n")

# LLM chain 생성
output_parser = LineListOutputParser()
llm_chain = custom_prompt | llm | output_parser

# multi-query retriever 생성
retriever_from_llm = MultiQueryRetriever(retriever=vectorstore.as_retriever(), llm_chain=llm_chain, parser_key="lines") # 사용자의 질문을 여러 관점에서 재구성하여 다양한 쿼리를 생성
retriever_from_llm.verbose = True

# 쿼리 실행
query = "mama mia?"
results = retriever_from_llm.get_relevant_documents(query)

# 결과 출력
for i, doc in enumerate(results[:5]):  # 상위 5개 결과 출력
    print(f"문서 {i+1}:")
    print(doc.page_content + "\n")






