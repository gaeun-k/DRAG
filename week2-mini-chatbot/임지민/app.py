"""Local-only study RAG. Run with python -m streamlit run app.py."""
import hashlib
import json
from pathlib import Path

import streamlit as st
from advanced_rag import Options, StudyRag, pretty_text
from ingest import load_file, load_url
from local_ai import model_status, describe_image
from runtime_policy import MAX_FILE_MB, MAX_TOTAL_MB, validate_upload_sizes

ROOT = Path(__file__).resolve().parent
APP_VERSION = "local-rag-2"
st.set_page_config(page_title="나의 문서 · RAG 스터디", page_icon="📚", layout="wide")
st.title("나의 문서, 근거 있는 답변")
st.caption("2주차 · LangChain 미니 RAG | 문서 → 청크 → 검색 → 근거 있는 답변")


@st.cache_data(ttl=15)
def ai_ready():
    return model_status()


def show_sources(sources):
    if not sources:
        return
    st.caption(f"참고한 자료 {len(sources)}개 · 펼치면 원문과 출처를 확인할 수 있습니다.")
    for source in sources:
        name = source["source"]
        short = name if len(name) <= 48 else name[:45] + "…"
        page = f" · {source['page']}페이지" if "page" in source else ""
        kind = {"image": "이미지 분석", "table": "표", "text": "본문"}.get(source.get("kind"), "본문")
        with st.expander(f"[{source['number']}] {short}{page} · {kind}"):
            st.caption(name)
            if source.get("excerpt"):
                st.caption("모델 입력 길이에 맞춰 이 청크의 앞부분을 발췌했습니다.")
            if source.get("url"):
                st.link_button("원문 페이지 열기", source["url"])
            if source.get("is_description"):
                st.caption("로컬 AI의 이미지 설명입니다. 작은 글자와 수치는 아래 원본과 비교하세요.")
            st.markdown(pretty_text(source["text"]))
            if source.get("image_path") and Path(source["image_path"]).is_file():
                st.image(source["image_path"], caption="분석에 사용한 원본 페이지/이미지")


def show_answer(message):
    st.markdown(message["content"])
    if message.get("citation_warning"):
        st.warning("답변의 일부 근거 번호가 올바르지 않습니다. 원문과 비교해 주세요.")
    if "sources" in message:
        st.divider()
        show_sources(message["sources"])


ready_ai = ai_ready()
with st.sidebar:
    st.header("1. 문서 준비")
    st.success("무료 · 내 컴퓨터에서 처리")
    mode_label = st.selectbox("답변 방식", ["로컬 AI 답변", "원문 검색만"], index=0 if ready_ai else 1)
    mode = "local" if mode_label == "로컬 AI 답변" else "demo"
    if mode == "local":
        st.caption("API 키·결제 없이 답변합니다. 이전 대화도 참고합니다.")
        if not ready_ai:
            st.warning("run_local_ai.bat 실행 후 잠시 기다리고 새로고침하세요.")
    use_sample = st.checkbox("실습용 자체 위키 사용", value=True)
    use_folder = st.checkbox("data 폴더의 문서도 함께 읽기", value=False)
    uploads = st.file_uploader("내 위키·블로그·문서", type=["txt", "md", "html", "htm", "pdf", "csv", "py", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    st.caption(f"파일당 {MAX_FILE_MB}MB, 총 {MAX_TOTAL_MB}MB(1GB)까지.")
    pasted = st.text_area("문서 본문 직접 붙여넣기", height=100, max_chars=100000)
    pasted_name = st.text_input("붙여넣은 문서 제목", value="내 블로그")
    urls = st.text_area("웹 문서 주소 (한 줄에 하나, 최대 5개)", height=70)
    st.caption("로그인 없이 볼 수 있는 글만 수집합니다. 주소를 바꾼 뒤 인덱스를 다시 만드세요.")
    vision = st.checkbox("이미지·표 시각 분석", value=False, disabled=mode != "local")
    visual_pages = st.number_input("PDF마다 앞에서 분석할 페이지 수", 1, 100, 3, disabled=not vision)
    if vision:
        st.caption("텍스트와 추출 가능한 표는 전체 페이지에서 읽습니다. 시각 분석은 지정한 앞쪽 페이지에만 적용하며 시간이 더 걸립니다.")
    st.header("2. 청킹과 검색")
    chunk_size = st.slider("청크 크기(문자)", 100, 3000, 1000, 100)
    overlap = st.number_input("겹침(문자)", 0, chunk_size - 1, min(100, chunk_size - 1), 10)
    k = st.slider("검색할 청크 수 k", 1, 8, 4)
    searches = {"혼합 검색 (BM25 + 의미)": "hybrid", "의미 검색": "similarity", "단어 검색 (BM25)": "bm25",
                "중복을 줄이는 검색 (MMR)": "mmr", "관련성 기준 검색": "threshold"}
    if mode == "local": searches["여러 질문으로 검색 (MultiQuery)"] = "multiquery"
    search = searches[st.selectbox("검색 방식", list(searches))]
    with st.expander("심화 실험 설정"):
        splitters = {"문단 우선 재귀 분할": "recursive", "구분자 기준 분할": "character"}
        if mode == "local": splitters["의미 기준 분할"] = "semantic"
        splitter = splitters[st.selectbox("청킹 방식", list(splitters))]
        if splitter == "semantic": st.caption("의미 분할은 겹침 대신 내용의 의미 변화와 최대 길이로 경계를 정합니다.")
        store = st.selectbox("벡터 저장소", ["faiss", "chroma"])
        threshold = st.number_input("관련성 기준값", 0.0, 1.0, 0.35, 0.05)
        raptor = st.checkbox("RAPTOR 계층 요약 검색", disabled=mode != "local")
        if raptor: st.caption("최대 120개 청크·3개 층. 원문과 계층 요약을 함께 검색합니다.")
    build = st.button("문서 인덱스 만들기", type="primary", use_container_width=True)
    clear = st.button("대화 지우기", use_container_width=True)

options = Options(mode, chunk_size, int(overlap), k, search, splitter, store, threshold, raptor)
files = []
if use_sample: files.append(("study_wiki.md", (ROOT / "data/study_wiki.md").read_bytes()))
if use_folder:
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".csv", ".html", ".py"}:
            if use_sample and path.name == "study_wiki.md":
                continue
            if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
                st.error(f"{path.name}: 30MB를 초과합니다.")
                st.stop()
            files.append((str(path.relative_to(ROOT / "data")), path.read_bytes()))
files.extend((f.name, f.getvalue()) for f in uploads)
if pasted.strip(): files.append(((pasted_name.strip() or "붙여넣은 문서") + ".txt", pasted.encode()))
url_list = [u.strip() for u in urls.splitlines() if u.strip()]
fingerprint = hashlib.sha256((APP_VERSION + repr(options) + repr(url_list) + str((vision, visual_pages))).encode())
for name, content in files:
    fingerprint.update(repr((name, len(content))).encode())
    fingerprint.update(content)
signature = fingerprint.hexdigest()
st.session_state.setdefault("messages", [])
if clear:
    st.session_state.messages = []
    if hasattr(st.session_state.get("bot"), "clear_history"): st.session_state.bot.clear_history()

if build:
    st.session_state.pop("bot", None)
    st.session_state.pop("density", None)
    st.session_state.messages = []
    try:
        if not files and not url_list: raise ValueError("문서를 올리거나 본문/주소를 넣어주세요.")
        if len(url_list) > 5: raise ValueError("웹 주소는 한 번에 5개까지 넣어주세요.")
        validate_upload_sizes(len(content) for _, content in files)
        if mode == "local" and not ready_ai: raise ValueError("로컬 AI가 준비되지 않았습니다. run_local_ai.bat를 실행하세요.")
        with st.status("문서를 읽고 검색을 준비합니다…", expanded=True) as status:
            documents = []
            for name, content in files:
                status.write(f"읽는 중: {name}")
                documents.extend(load_file(name, content, vision, int(visual_pages), describe_image))
            for url in url_list:
                status.write(f"웹 문서 읽는 중: {url}")
                documents.extend(load_url(url))
            bot = StudyRag(documents, options, progress=status.write)
            status.update(label=f"{len(bot.chunks)}개 청크 준비 완료", state="complete", expanded=False)
        st.session_state.bot = bot
        st.session_state.signature = signature
    except Exception as exc:
        st.error(str(exc) if isinstance(exc, ValueError) else f"처리에 실패했습니다({type(exc).__name__}). 파일 또는 웹 주소를 확인하세요.")

bot = st.session_state.get("bot")
ready = bot is not None and st.session_state.get("signature") == signature
if mode == "demo":
    st.info("원문 검색 모드입니다. 짧은 발췌를 보여주며 새 답변을 생성하지 않습니다. 로컬 AI 답변 모드는 무료입니다.")
if bot is not None and not ready:
    st.warning("문서·설정 또는 프로그램이 바뀌었습니다. 인덱스를 다시 만들어 주세요.")
if not bot:
    st.markdown("왼쪽에서 **문서 인덱스 만들기**를 누른 뒤 질문하세요.")
    st.markdown("예시: `2주차 과제 제출물은 무엇인가요?` · `역할 분담은 어떻게 하나요?`")
if ready:
    a, b, c = st.columns(3)
    a.metric("검색 대상 청크", len(bot.chunks))
    b.metric("청크 크기", chunk_size)
    c.metric("검색 개수", k)
    with st.expander("청킹 결과 살펴보기"):
        for chunk in bot.chunks[:30]:
            st.caption(f"청크 {chunk.metadata['chunk_id']} · {chunk.metadata['source']} · {len(chunk.page_content)}자")
            st.markdown(pretty_text(chunk.page_content))
        if len(bot.chunks) > 30: st.caption("처음 30개 청크만 표시합니다.")
    with st.expander("학습 실험 · 검색 비교와 밀도 요약"):
        compare_query = st.text_input("비교할 검색 질문", value="과제 제출물은 무엇인가요?")
        if st.button("검색 방식 비교"):
            for strategy in ["bm25", "similarity", "hybrid", "mmr"]:
                hits = bot.resolve(bot.retrieve(compare_query, strategy))
                st.markdown(f"**{strategy}**")
                st.write([f"{d.metadata['source']} · 청크 {d.metadata['chunk_id']}" for d in hits])
        if st.button("5단계 밀도 요약 만들기", disabled=mode != "local"):
            try:
                with st.spinner("빠진 정보를 보완하며 5번 요약합니다…"):
                    st.session_state.density = bot.density_summary()
            except ValueError as exc: st.error(str(exc))
        for i, summary in enumerate(st.session_state.get("density", []), 1):
            st.markdown(f"**{i}단계**\n\n{summary}")

st.caption("로컬 AI는 최근 3회 대화를 참고합니다. 문서·설정을 바꾸거나 대화를 지우면 기억도 초기화됩니다.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        show_answer(message) if message["role"] == "assistant" else st.markdown(message["content"])

question = st.chat_input("문서에 대해 질문해 주세요", disabled=not ready, max_chars=2000)
if question:
    with st.chat_message("user"): st.markdown(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("문서 근거를 확인하고 답변을 정리합니다…"):
                result = bot.ask(question)
            message = {"role": "assistant", "content": result["answer"], **{key: value for key, value in result.items() if key != "answer"}}
            show_answer(message)
        st.session_state.messages.extend([{"role": "user", "content": question}, message])
    except Exception as exc:
        st.error(str(exc) if isinstance(exc, ValueError) else f"응답 생성에 실패했습니다({type(exc).__name__}). 잠시 후 다시 시도하세요.")
if st.session_state.messages:
    st.download_button("대화와 근거 기록 저장", json.dumps({"settings": options.__dict__, "messages": st.session_state.messages}, ensure_ascii=False, indent=2),
                       file_name="rag-study-record.json", mime="application/json")
