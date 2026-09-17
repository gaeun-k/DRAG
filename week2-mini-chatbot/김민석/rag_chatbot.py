"""
TAVE 18기 DRAG 스터디 - 2주차 과제
자체 문서 기반 미니 RAG 챗봇

Base: teddynote(langchain-kr) 12-RAG/01-RAG-Basic-Webloader.ipynb

원본은 네이버 뉴스 기사 한 건을 고정된 URL로, OpenAI 임베딩/LLM으로 QA를 수행하는
예제였다. 아래와 같이 수정했다.
  1) 네이버 뉴스 URL + bs4.SoupStrainer(뉴스 전용 태그 필터)
     -> 원하는 위키/블로그 URL을 그대로 로드하도록 변경 (사이트마다 태그 구조가
        달라서 특정 class를 골라내는 필터는 빼고 페이지 전체를 가져온다)
  2) 미리 정해둔 질문 4개를 순서대로 스트림 출력
     -> 터미널에서 원하는 질문을 계속 입력하는 챗봇 루프로 변경
  3) LangSmith 로깅(langchain_teddynote.logging)은 필수가 아니라 제외
  4) OpenAI LLM -> Google Gemini(무료 API 키)로 교체
     (OpenAI는 결제 등록이 필요해서, 무료로 바로 테스트할 수 있는 Gemini로 변경)
  5) 임베딩은 Google 임베딩 API가 프로젝트 할당량 문제(429)로 계속 막혀서,
     로컬에서 무료로 돌아가는 HuggingFace 한국어 임베딩 모델로 교체
     (인터넷/할당량과 무관하게 내 컴퓨터에서 직접 계산)

문서 분할 / 벡터스토어(FAISS) 생성 / 검색기 / 프롬프트 / 체인 구성은
원본 구조를 그대로 따른다.
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

# API 키(.env의 GOOGLE_API_KEY)를 로드
load_dotenv()


# ------------------------------------------------------------------
# 0. 설정 - 원본은 네이버 뉴스 URL 하나로 고정되어 있던 부분을
#    본인이 원하는 위키/블로그 문서로 바꿀 수 있게 변수로 뺐다.
# ------------------------------------------------------------------
SOURCE_URL = "https://ko.wikipedia.org/wiki/인공지능"  # 원하는 위키/블로그 주소로 교체


# ------------------------------------------------------------------
# 1단계: 문서 로드 (Document Load)
#   특정 사이트(예: 네이버 뉴스처럼 본문 class가 정해진 곳)를 쓸 경우엔
#   원본처럼 bs_kwargs=dict(parse_only=bs4.SoupStrainer(...))를 추가해서
#   본문만 뽑아낼 수도 있다. 여기서는 범용성을 위해 페이지 전체를 가져온다.
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
#   Google 임베딩 API가 프로젝트 할당량 문제(429)로 계속 막혀서,
#   로컬에서 돌아가는 한국어 임베딩 모델(ko-sroberta-multitask)을 사용한다.
#   최초 실행 시 모델 파일(약 400MB)을 한 번 내려받고, 이후로는 인터넷
#   연결 없이도 로컬에서 계산된다.
# ------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
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
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)


# ------------------------------------------------------------------
# 7단계: LLM
#   gemini-2.5-flash는 신규 사용자에게 더 이상 제공되지 않아
#   gemini-3.6-flash로 교체했다.
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)


# ------------------------------------------------------------------
# 8단계: 체인 생성
# ------------------------------------------------------------------
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ------------------------------------------------------------------
# 실행부 - 원본은 질문 4개를 코드에 하드코딩해서 스트림 출력했지만,
#    여기서는 터미널에서 계속 질문을 입력받는 챗봇 형태로 바꿨다.
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

        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    chat()