"""
TAVE 18기 DRAG 스터디 - 2주차 과제
나무위키 검색 기반 미니 RAG 챗봇

Base: teddynote(langchain-kr) 12-RAG/01-RAG-Basic-Webloader.ipynb

rag_chatbot.py(위키백과 고정 문서 버전) 대비 변경 사항
  - 시작할 때 문서 1개를 고정으로 로드하는 대신, 매 턴마다 "단어"를 입력받아
    그 단어의 위키 문서(https://namu.wiki/w/단어)를 새로 크롤링 -> 분할 -> 임베딩 -> 검색 인덱스 구축
  - 이어서 "질문"을 입력받아, 방금 만든 그 단어의 문서를 근거로 답변
  - 즉 한 턴 = (단어, 질문) 한 쌍. 다른 단어를 검색하면 인덱스를 새로 만든다.

임베딩(BAAI/bge-m3, 로컬)·LLM(gemini-3.8-flash, 무료 API) 조합은 rag_chatbot.py와 동일.
"""

from urllib.parse import quote

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
# 임베딩 모델은 한 번만 로드해서 재사용 (턴마다 새로 로드하면 느림)
# ------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},  # cuda, cpu
    encode_kwargs={"normalize_embeddings": True},
)

prompt = PromptTemplate.from_template(
    """당신은 위키 문서를 근거로 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 답변을 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""
)

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)


def build_chain(word: str):
    """주어진 단어의 위키 문서를 로드해서 그 문서 전용 RAG 체인을 만든다."""
    url = f"https://namu.wiki/w/{quote(word)}"
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    print(f"[{word}] 위키 문서 로드 완료 (문서 수: {len(docs)})")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"[{word}] 분할된 청크 수: {len(splits)}")

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def chat():
    print("\n=== 나무위키 검색 챗봇 (종료: exit, 다른 단어 검색: 새단어) ===")
    while True:
        word = input("\n검색할 단어: ").strip()
        if word.lower() in ("exit", "quit"):
            print("챗봇을 종료합니다.")
            break
        if not word:
            continue

        try:
            rag_chain = build_chain(word)
        except Exception as e:
            print(f"'{word}' 문서를 불러오지 못했습니다: {e}")
            continue

        # 같은 문서에 대해 exit 하기 전까지 계속 질문을 받는다.
        # "새단어"를 입력하면 이 문서는 그만 묻고 다른 단어를 다시 검색한다.
        print(f"['{word}' 문서 로드 완료. 이제 이 문서에 대해 계속 질문하세요]")
        while True:
            question = input("\n질문: ").strip()
            if question.lower() in ("exit", "quit"):
                print("챗봇을 종료합니다.")
                return
            if question in ("새단어", "새 단어"):
                break
            if not question:
                continue

            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print()


if __name__ == "__main__":
    chat()
