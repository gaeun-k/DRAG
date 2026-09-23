"""
TAVE 18기 DRAG 스터디 - 2주차 과제
자체 문서 기반 미니 RAG 챗봇

Base: teddynote(langchain-kr) 12-RAG/01-RAG-Basic-Webloader.ipynb

원본 대비 변경 사항
  1) 네이버 뉴스 고정 URL -> 원하는 위키/블로그 URL(SOURCE_URL)로 교체 가능하게 변경
  2) 미리 정해둔 질문 4개를 순서대로 스트림 출력 -> 터미널에서 계속 질문을 입력하는 챗봇 루프로 변경
  3) LangSmith 로깅은 필수가 아니라 제외
  4) 임베딩/LLM 모두 유료 API(OpenAI)가 막혀서 무료 조합으로 교체
     - 임베딩: OpenAIEmbeddings -> HuggingFaceEmbeddings(BAAI/bge-m3) (로컬 계산, 완전 무료. 1주차 CH08에서 다룬 모델을 그대로 재사용)
     - LLM: ChatOpenAI -> ChatGoogleGenerativeAI(gemini-3.8-flash) (무료 API 키)

문서 분할 / 벡터스토어(FAISS) 생성 / 검색기 / 프롬프트 / 체인 구성은 원본 구조를 그대로 따른다.
"""

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# ------------------------------------------------------------------
# 0. 설정 - 원하는 위키/블로그 문서 주소로 바꿔서 사용
# ------------------------------------------------------------------
SOURCE_URL = "https://ko.wikipedia.org/wiki/인공지능"  # 원하는 위키/블로그 주소로 교체


# ------------------------------------------------------------------
# 1단계: 문서 로드 (Document Load)
# ------------------------------------------------------------------
loader = WebBaseLoader(web_paths=(SOURCE_URL,))
docs = loader.load()
print(f"문서의 수: {len(docs)}")


# ------------------------------------------------------------------
# 2단계: 분할 (Text Split)
# ------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"분할된 청크 수: {len(splits)}")


# ------------------------------------------------------------------
# 3~4단계: 임베딩 + 벡터DB(FAISS) 저장
#   로컬에서 도는 다국어 임베딩 모델을 사용 (API 키/할당량 무관)
# ------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},  # cuda, cpu
    # 벡터 길이를 1로 정규화 -> 코사인 유사도 계산 더 안정적
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)


# ------------------------------------------------------------------
# 5단계: 검색기 (Retriever)
# ------------------------------------------------------------------
retriever = vectorstore.as_retriever()


# ------------------------------------------------------------------
# 6단계: 프롬프트 (원본 프롬프트를 그대로 사용)
# ------------------------------------------------------------------
prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 답변을 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""
)


# ------------------------------------------------------------------
# 7단계: LLM
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)


# ------------------------------------------------------------------
# 8단계: 체인 생성
# ------------------------------------------------------------------
rag_chain = (
    # RunnablePassthrough() : "입력을 그대로 통과" 시키는 역할
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    # StrOutputParser() : llm의 반환(메세지)에서 순수 텍스트만 추출
    | StrOutputParser()
)


# ------------------------------------------------------------------
# 실행부 - 터미널에서 계속 질문을 입력받는 챗봇 루프
# ------------------------------------------------------------------
def chat():
    print("\n=== 미니 RAG 챗봇 (종료하려면 exit 입력) ===")
    while True:
        question = input("\n질문: ").strip()
        if question.lower() in ("exit", "quit"):
            print("챗봇을 종료합니다.")
            break
        if not question:
            continue
        
        # 토큰이 생성되는 대로 조금씩 잘라서 실시간으로 반환
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    chat()
