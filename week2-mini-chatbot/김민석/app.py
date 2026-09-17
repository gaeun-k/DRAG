"""
TAVE 18기 DRAG 스터디 - 2주차 과제 (커스텀 버전)
자체 문서 기반 미니 RAG 챗봇 - Streamlit 웹 UI + 대화 기억 + 멀티 문서 버전

Base: rag_chatbot.py (teddynote 12-RAG/01-RAG-Basic-Webloader.ipynb를 수정한 버전)

rag_chatbot.py(터미널 버전)에 아래 기능들을 얹었다.

  1) 이전 대화를 기억해서 후속 질문도 이해하도록
     - 질문을 검색하기 전에, LLM이 이전 대화를 참고해서 "독립적인 질문"으로
       한 번 다시 써주는 단계를 추가했다.
       예) "그건 누가 만들었어?" -> "인공지능은 누가 만들었어?"

  2) 터미널 대신 Streamlit으로 웹페이지 채팅 UI 구성
     - 문서 로드 + 임베딩 + 벡터스토어 구축은 @st.cache_resource로 캐싱해서,
       같은 URL 목록이면 다시 계산하지 않는다.

  3) 문서 하나만 학습하는 한계를 풀기 위해, 여러 URL을 한 번에 넣어서
     하나의 벡터스토어로 합칠 수 있게 했다 (WebBaseLoader는 원래 URL 여러 개를
     한 번에 받을 수 있어서, 사이드바 입력만 여러 줄로 바꿨다).

  4) 문서에 없는 질문이면 무조건 "모른다"고 답하던 것을, 문서 내용을
     우선하되 문서에 없으면 AI의 일반 지식으로도 답하도록(대신 그 사실을
     먼저 밝히도록) 선택할 수 있게 했다. 사이드바 체크박스로 켜고 끌 수 있다.

실행 방법: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # .env의 GOOGLE_API_KEY 로드

DEFAULT_SOURCE_URLS = "https://ko.wikipedia.org/wiki/인공지능"


# ------------------------------------------------------------------
# 문서 로드 + 분할 + 임베딩 + 벡터스토어 구축
#   URL 목록이 바뀔 때만 다시 계산하도록 st.cache_resource로 캐싱한다.
#   WebBaseLoader(web_paths=...)는 URL을 여러 개 튜플로 받을 수 있어서,
#   여러 문서를 하나의 벡터스토어로 한 번에 합칠 수 있다.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_retriever(source_urls: tuple[str, ...]):
    loader = WebBaseLoader(web_paths=source_urls)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(), len(docs), len(splits)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(history, max_turns=5):
    if not history:
        return "(이전 대화 없음)"
    lines = []
    for q, a in history[-max_turns:]:
        lines.append(f"사용자: {q}")
        lines.append(f"챗봇: {a}")
    return "\n".join(lines)


# 후속 질문을 이전 대화 없이도 이해 가능한 독립 질문으로 재작성하는 프롬프트
CONTEXTUALIZE_PROMPT = PromptTemplate.from_template(
    """아래는 사용자와 챗봇의 이전 대화 기록입니다. 대화 흐름을 참고해서,
마지막 사용자 질문이 이전 대화 없이도 이해할 수 있는 독립적인 질문이 되도록
다시 작성하세요. 이미 독립적인 질문이면 그대로 반환하세요. 질문만 출력하고
다른 설명은 덧붙이지 마세요.

[이전 대화]
{chat_history}

[마지막 질문]
{question}

[독립적인 질문]"""
)

# 문서(context)에서만 답하는 프롬프트 (원본 방식)
STRICT_ANSWER_PROMPT = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context)과 이전 대화(chat_history)를 참고하여 질문(question)에 답하는 것입니다.
검색된 다음 문맥(context)을 사용하여 질문(question)에 답하세요. 만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다`라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

[이전 대화]
{chat_history}

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# 문서에 없으면 일반 지식으로도 답하는 프롬프트 (확장 버전)
HYBRID_ANSWER_PROMPT = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context)과 이전 대화(chat_history)를 참고하여 질문(question)에 답하는 것입니다.
검색된 문맥(context)에 관련 내용이 있으면 그것을 우선적으로 사용해서 답하세요.
문맥에서 답을 찾을 수 없는 질문이면, 당신이 알고 있는 일반적인 지식으로 답하되,
답변 맨 앞부분에 "(문서에는 없는 내용이라 일반 지식으로 답할게요)"라고 먼저 밝히고 이어서 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

[이전 대화]
{chat_history}

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)


# ------------------------------------------------------------------
# Streamlit 화면 구성
# ------------------------------------------------------------------
st.set_page_config(page_title="미니 RAG 챗봇", page_icon="📚")
st.title("📚 미니 RAG 챗봇")
st.caption("자체 문서를 기반으로 답하는 RAG 챗봇 (TAVE DRAG 2주차 과제)")

with st.sidebar:
    st.header("설정")
    source_urls_text = st.text_area(
        "문서 URL (한 줄에 하나씩, 여러 개 가능)",
        value=DEFAULT_SOURCE_URLS,
        height=120,
    )
    use_general_knowledge = st.checkbox(
        "문서에 없으면 AI 일반 지식으로도 답하기", value=True
    )
    load_clicked = st.button("이 문서들로 불러오기")

    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = None

    urls = tuple(u.strip() for u in source_urls_text.splitlines() if u.strip())

    if load_clicked or st.session_state.loaded_urls is None:
        if not urls:
            st.warning("URL을 하나 이상 입력해줘.")
        else:
            with st.spinner("문서를 불러오고 벡터DB를 구축하는 중... (처음엔 조금 걸려요)"):
                retriever, n_docs, n_chunks = build_retriever(urls)
            st.session_state.retriever = retriever
            st.session_state.loaded_urls = urls
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success(f"문서 {n_docs}개 / 청크 {n_chunks}개 로드 완료")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 새 질문 입력
question = st.chat_input("문서에 대해 궁금한 걸 물어보세요")

if question:
    if "retriever" not in st.session_state:
        st.warning("먼저 왼쪽에서 문서를 불러와줘.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    llm = get_llm()

    # 1) 이전 대화를 참고해서 독립적인 질문으로 재작성
    history_text = format_chat_history(st.session_state.chat_history)
    if st.session_state.chat_history:
        standalone_question = (
            (CONTEXTUALIZE_PROMPT | llm | StrOutputParser())
            .invoke({"chat_history": history_text, "question": question})
            .strip()
        )
    else:
        standalone_question = question

    # 2) 재작성된 질문으로 문서 검색
    docs = st.session_state.retriever.invoke(standalone_question)
    context = format_docs(docs)

    # 3) 문맥 + 이전 대화 + 원래 질문으로 답변 생성 (스트리밍)
    answer_prompt = HYBRID_ANSWER_PROMPT if use_general_knowledge else STRICT_ANSWER_PROMPT
    answer_chain = answer_prompt | llm | StrOutputParser()
    with st.chat_message("assistant"):
        answer = st.write_stream(
            answer_chain.stream(
                {"chat_history": history_text, "question": question, "context": context}
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append((question, answer))