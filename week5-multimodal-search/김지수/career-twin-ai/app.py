"""실행: python -m streamlit run app.py"""
import hashlib
import hmac
import json
import os
from pathlib import Path

import streamlit as st

from career_twin.config import SETTINGS
from career_twin.dataset import load_samples
from career_twin.ingest import extract, render_page
from career_twin.local_ai import Ollama
from career_twin.retrieval import search, source_label
from career_twin.services import (coverage, experience_answer, explain_paper,
                                 extract_requirements, judge_requirement, recommend_jobs)
from career_twin.storage import Store

st.set_page_config(page_title="Career Twin AI", page_icon="🌱", layout="wide")
st.markdown("""<style>
.stApp {background:#eff8ff;color:#1f2937;}
.block-container {max-width:1180px;padding-top:2rem;}
h1,h2,h3 {color:#173b66;}
div.stButton > button[kind="primary"] {background:#173b66;border:0;border-radius:12px;}
[data-testid="stMetric"] {background:white;border-radius:16px;padding:16px;}
</style>""", unsafe_allow_html=True)


def authenticate():
    password = os.getenv("APP_PASSWORD", "")
    required = os.getenv("APP_REQUIRE_AUTH", "false").lower() == "true"
    if required and not password:
        st.error("배포 환경에서는 APP_PASSWORD를 설정해야 합니다.")
        st.stop()
    if password and not st.session_state.get("authenticated"):
        st.title("Career Twin AI 로그인")
        entered = st.text_input("개인 작업 공간 비밀번호", type="password")
        if st.button("로그인", type="primary"):
            if hmac.compare_digest(entered, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 일치하지 않습니다.")
        st.stop()


authenticate()
store = Store(SETTINGS.data_dir / "private")


def clear_results():
    for key in ("experience_result", "match_results", "paper_hits", "paper_result", "job_hits"):
        st.session_state.pop(key, None)


def display_hits(hits):
    for hit in hits:
        with st.expander(source_label(hit)):
            st.write(hit["text"])
            st.caption(f"청크 ID: {hit['id']} · 검색: {hit.get('search_method', '')}")


with st.sidebar:
    st.header("실행 설정")
    mode = st.radio("답변 방식", ["검색 전용", "Ollama AI 답변"])
    semantic = st.checkbox("의미 검색 + FAISS", value=False,
                           help="Ollama embeddinggemma 필요. BM25와 RRF로 합칩니다.")
    st.caption("검색 전용: 키워드 검색·원문 확인 가능. AI 답변과 역량 판정은 실행하지 않습니다.")
    if st.button("Ollama 연결 확인"):
        try:
            st.write(Ollama().models())
        except Exception as exc:
            st.error(str(exc))
    st.caption(f"텍스트: {SETTINGS.chat_model}\n\n임베딩: {SETTINGS.embed_model}\n\nVLM: {SETTINGS.vision_model}")
    st.caption("자료는 이 앱 서버에 저장됩니다. 기본 설정은 개인용 단일 작업 공간입니다.")
    if os.getenv("APP_PASSWORD") and st.button("로그아웃"):
        st.session_state.clear()
        st.rerun()

ai = Ollama() if mode == "Ollama AI 답변" else None
retrieval_ai = Ollama() if semantic else None
st.title("🌱 Career Twin AI")
st.write("내 경험을 근거로 직무를 탐색하고, 논문으로 면접을 준비하세요.")
st.caption("공개 사례는 합성 실험 자료입니다. 실제 채용 공고·합격자 사례·합격 확률을 제공하지 않습니다.")
documents_tab, jobs_tab, experience_tab, matching_tab, papers_tab = st.tabs([
    "내 자료", "공고 탐색", "경험 질문", "JD 근거 매칭", "논문 · 면접 준비"])

with documents_tab:
    st.subheader("개인 문서와 논문 등록")
    st.write("개인 경험 자료와 논문은 별도 검색 공간에 저장합니다.")
    with st.form("upload_form", clear_on_submit=True):
        kind_label = st.selectbox("자료 구분", ["내 경험 문서", "논문 · 학술자료"])
        uploads = st.file_uploader("PDF · Markdown · TXT", type=["pdf", "md", "txt"], accept_multiple_files=True)
        submitted = st.form_submit_button("문서 등록", type="primary")
    if submitted and uploads:
        for upload in uploads:
            try:
                content = upload.getvalue()
                chunks, warnings = extract(upload.name, content)
                store.add(upload.name, content, "personal" if kind_label == "내 경험 문서" else "paper", chunks, warnings)
                st.success(f"{upload.name}: {len(chunks)}개 청크 등록")
                for warning in warnings:
                    st.warning(warning)
            except Exception as exc:
                st.error(f"{upload.name}: {exc}")
        clear_results()
    st.caption("데모 문서는 examples/에 있습니다. 자기 경험이 아닌 작성 예시임을 표시했습니다.")
    for doc in store.documents():
        left, right = st.columns([5, 1])
        left.write(f"{doc['name']} · {'개인 문서' if doc['kind'] == 'personal' else '논문'}")
        with left.expander("저장된 텍스트 보기"):
            selected_chunks = [c for c in store.chunks(doc["kind"]) if c["document_id"] == doc["id"]]
            for chunk in selected_chunks:
                st.caption(source_label(chunk))
                st.write(chunk["text"])
        if right.button("삭제", key="delete_" + doc["id"]):
            store.delete(doc["id"])
            clear_results()
            st.rerun()

with jobs_tab:
    st.subheader("Resume Matching Dataset v2 · 합성 공고 탐색")
    samples = load_samples()
    if not samples:
        st.info("공개 데이터가 아직 없습니다. 터미널에서 python scripts/prepare_dataset.py --limit 300 실행")
    st.caption(f"불러온 사례 {len(samples)}개. 내 문서 DB와 분리되어 있으며 AI 평가 점수는 검색에 사용하지 않습니다.")
    target = st.text_input("희망 직무·기술", placeholder="AI 개발 · Python · RAG · 벡터 검색")
    if st.button("내 경험으로 공고 찾기", type="primary"):
        personal = store.chunks("personal")
        if not target.strip():
            st.warning("희망 직무·기술을 입력하세요.")
        elif not personal:
            st.warning("개인 문서를 먼저 등록하세요.")
        else:
            try:
                relevant = search(target, personal, 5, retrieval_ai, store)
                query = target + "\n" + "\n".join(h["text"] for h in relevant)
                st.session_state.job_hits = recommend_jobs(query, samples)
            except Exception as exc:
                st.error(str(exc))
    for hit in st.session_state.get("job_hits", []):
        with st.container(border=True):
            st.write(hit["text"])
            st.caption(f"합성 공고 · BM25 검색 순위용 점수 {hit['score']:.3f} (적합도·확률 아님)")
            if st.button("이 공고의 요구사항 분석", key="choose_" + hit["id"]):
                st.session_state.jd_input = hit["text"]
                st.session_state.pop("requirements", None)
                st.session_state.pop("match_results", None)
                st.success("JD 근거 매칭 탭으로 이동하세요.")
            sample = next((s for s in samples if s["id"] == hit["id"]), None)
            if sample:
                with st.expander("같은 사례의 합성 지원서 보기 — 내 경험 아님"):
                    st.write(sample.get("resume", ""))
                    st.write(sample.get("selfintro", ""))
                    st.caption("데이터셋 속 생성 사례이며 합격자 지원서가 아닙니다.")

with experience_tab:
    st.subheader("내 경험에 질문하기")
    question = st.text_input("질문", placeholder="RAG 개발 직무에서 어떤 프로젝트를 강조할까?", key="experience_question")
    if st.button("경험 검색 · 답변", type="primary"):
        if not question.strip():
            st.warning("질문을 입력하세요.")
        elif len(question) > 1500:
            st.warning("질문은 1,500자 이하로 입력하세요.")
        else:
            try:
                with st.spinner("개인 문서에서 근거를 찾고 있습니다."):
                    hits = search(question, store.chunks("personal"), 4, retrieval_ai, store)
                    st.session_state.experience_result = experience_answer(question, hits, ai)
            except Exception as exc:
                st.error(str(exc))
    result = st.session_state.get("experience_result")
    if result:
        st.info(result["message"])
        for claim in result["claims"]:
            st.write(claim["text"])
            with st.expander(claim["source"]):
                st.write(claim["quote"])
        if result["rejected"]:
            st.warning(f"출처·원문 검증에 실패한 주장 {result['rejected']}개를 제외했습니다.")

with matching_tab:
    st.subheader("채용 요구사항과 내 경험 연결")
    jd = st.text_area("JD 본문", key="jd_input", height=220)
    if st.button("요구사항 추출", type="primary"):
        try:
            with st.spinner("요구사항을 정리합니다."):
                st.session_state.requirements = extract_requirements(jd, ai)
                st.session_state.jd_hash = hashlib.sha256(jd.encode()).hexdigest()
                st.session_state.pop("match_results", None)
                st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
        except Exception as exc:
            st.error(str(exc))
    requirements = st.session_state.get("requirements", [])
    if requirements:
        st.caption("추출된 문장을 수정·삭제하거나 행을 추가할 수 있습니다. 검색 전용 추출은 규칙 기반 초안입니다.")
        edited = st.data_editor(requirements, num_rows="dynamic", hide_index=True,
            column_config={"category": st.column_config.SelectboxColumn("구분", options=["필수", "우대", "업무"], required=True)},
            disabled=["id"], key=f"req_editor_{st.session_state.get('editor_version', 0)}", use_container_width=True)
        changed = st.session_state.get("jd_hash") != hashlib.sha256(jd.encode()).hexdigest()
        if changed:
            st.warning("JD가 변경됐습니다. 요구사항을 다시 추출하세요.")
        if st.button("요구사항별 근거 찾기", type="primary", disabled=changed):
            valid = [{"text": str(r.get("text") or "").strip(), "category": r.get("category"),
                      "quote": str(r.get("quote") or "")} for r in edited]
            valid = [r for r in valid if r["text"] and r["category"] in {"필수", "우대", "업무"}]
            if not store.chunks("personal"):
                st.warning("개인 문서를 먼저 등록하세요.")
            elif not valid or len(valid) > 12:
                st.warning("요구사항은 1~12개로 작성하세요.")
            else:
                try:
                    rows = []
                    progress = st.progress(0, text="요구사항별 검색·판정")
                    for i, requirement in enumerate(valid):
                        hits = search(requirement["text"], store.chunks("personal"), 3, retrieval_ai, store)
                        rows.append(judge_requirement(requirement, hits, ai))
                        progress.progress((i + 1) / len(valid))
                    st.session_state.match_results = rows
                except Exception as exc:
                    st.error(str(exc))
    rows = st.session_state.get("match_results", [])
    if rows and not (requirements and changed):
        st.dataframe([{k: row[k] for k in ("requirement", "category", "status", "source", "reason", "next_action")}
                      for row in rows], hide_index=True, use_container_width=True)
        metric = coverage(rows)
        if metric is not None:
            st.metric("필수 요구사항의 문서 근거 커버리지", f"{metric:.0%}")
            st.caption("직접 근거=1, 관련 경험=0.5, 근거 없음=0 / 필수 항목 수. 합격 확률이 아닙니다.")
        for row in rows:
            with st.expander(row["requirement"] + " · " + row["status"]):
                st.write(row["quote"] or row["reason"])
                display_hits(row.get("candidates", []))
        st.download_button("매칭 결과 JSON", json.dumps(rows, ensure_ascii=False, indent=2),
                           "career_match.json", "application/json")
        if st.button("이 요구사항으로 논문 준비"):
            st.session_state.paper_question = "다음 직무 요구사항의 핵심 개념을 설명하고 면접 연습 질문을 만들어줘: " + ", ".join(
                row["requirement"] for row in rows[:5])
            st.success("논문 · 면접 준비 탭으로 이동하세요.")

with papers_tab:
    st.subheader("논문 PDF 검색 · 아키텍처 해설")
    st.caption("업로드한 논문 안에서 검색합니다. 외부 논문 자동 수집·SigLIP 그림 검색은 후속 기능입니다.")
    question = st.text_input("논문 질문", key="paper_question", placeholder="SigLIP의 학습 구조와 CLIP의 차이는?")
    if st.button("논문 근거 검색", type="primary"):
        if not question.strip() or len(question) > 2000:
            st.warning("질문은 1~2,000자로 작성하세요.")
        else:
            try:
                st.session_state.paper_hits = search(question, store.chunks("paper"), 5, retrieval_ai, store)
                st.session_state.pop("paper_result", None)
            except Exception as exc:
                st.error(str(exc))
    hits = st.session_state.get("paper_hits", [])
    display_hits(hits)
    if not hits:
        st.info("논문 자료를 등록하고 관련 근거를 검색하세요.")
    papers = [d for d in store.documents("paper") if Path(d["path"]).suffix == ".pdf"]
    image = None
    extra = []
    if papers:
        selected = st.selectbox("설명할 PDF", papers, format_func=lambda d: d["name"])
        from pypdf import PdfReader
        count = len(PdfReader(selected["path"]).pages)
        preferred = next((h["page"] for h in hits if h["document_id"] == selected["id"] and h.get("page")), 1)
        page = st.number_input("그림이 있는 페이지", min_value=1, max_value=count, value=preferred,
                               key="page_" + selected["id"] + "_" + str(preferred))
        try:
            image = render_page(selected["path"], int(page))
            st.image(image, caption=f"{selected['name']} · {page}쪽", width=650)
            extra = [c for c in store.chunks("paper") if c["document_id"] == selected["id"] and c["page"] == page]
        except Exception as exc:
            st.error(str(exc))
    use_vision = st.checkbox("이 페이지 그림도 VLM으로 읽기", value=False, disabled=image is None)
    if st.button("논문 설명 · 면접 연습 질문", type="primary", disabled=not hits):
        try:
            # 선택한 페이지의 본문을 우선 전달하여 그림·본문의 연결을 유지합니다.
            context = list({h["id"]: h for h in (extra[:2] + hits) if h}.values())[:4] if use_vision else hits[:3]
            with st.spinner("그림·문서 설명을 생성합니다. 첫 모델 로딩은 시간이 걸릴 수 있습니다."):
                answer = explain_paper(question, context, ai, image if use_vision else None)
                st.session_state.paper_result = {"answer": answer, "sources": [source_label(h) for h in context]}
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get("paper_result"):
        st.markdown(st.session_state.paper_result["answer"])
        st.write("전달한 근거:", st.session_state.paper_result["sources"])
        st.caption("VLM 설명은 생성 결과입니다. 특히 작은 글자·화살표 해석과 인용을 원문에서 확인하세요.")
