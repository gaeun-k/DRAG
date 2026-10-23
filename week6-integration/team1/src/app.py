"""
알약 정보 챗봇 데모 (Gradio)

코랩에서 실행
    import sys, os
    PROJECT = "/content/drive/MyDrive/프로젝트폴더"      # identify.py, rag.py, app.py 가 있는 폴더
    BASE = f"{PROJECT}/pill_chatbot"                     # master.parquet, index/ 가 있는 폴더
    sys.path.append(PROJECT)
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

    import app
    app.launch(BASE, share=True)

흐름
    [기능 1] 사진 업로드 → 약 찾기 → 후보 3개 → 하나 선택 → 약 정보 보기 → (이어서 그 약에 대해 질문)
    [기능 2] 채팅창에 증상 입력 → 관련 일반의약품 정보
"""

import os
import traceback
import zipfile

import gradio as gr

import identify
import rag

# ------------------------------------------------------------------ 설정
INFO_QUESTION = "이 약의 이름, 종류, 복용법, 주의사항을 알려줘"
USE_TEMPLATE_MODE = False   # 팀원이 search(question=None, item_seq=...) 템플릿 모드를 만들면 True로
IMPRINT_EVIDENCE = 0.8      # 각인이 이 이상 맞으면 '근거 있음'으로 봄 (low_confidence 판단용, 임시)

DISCLAIMER = ("⚠️ 이 정보는 식품의약품안전처 공개 자료를 바탕으로 한 **참고용**이며, 의학적 진단이나 처방이 아닙니다. "
              "복용 전 반드시 약사·의사와 상담하세요. 국내 일반의약품만 지원합니다.")

# ------------------------------------------------------------------ 디자인 (여기만 고쳐도 화면이 바뀜)
THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Noto Sans KR"), "sans-serif"])

CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
#title h1 { margin-bottom: 0.2em; }
#disclaimer { background: #fff7e6; border: 1px solid #ffd591; border-radius: 10px; padding: 8px 14px; font-size: 0.9em; }
#status { min-height: 2.5em; }
#find-btn, #info-btn { font-size: 1.05em; }
"""

_loaded = {"ok": False}
LOCAL_IMG_DIR = "/content/pill_images"   # DB 알약 사진을 풀어둘 위치


# ------------------------------------------------------------------ 로드
def load(base_dir):
    """모델은 한 번만 불러옴. app.py를 다시 불러와도(디자인 수정 후) 모델은 다시 안 불러옴."""
    if not identify.S.get("index"):
        identify.load(base_dir)
    _prepare_local_images(base_dir)
    if hasattr(rag, "load") and not getattr(rag, "_app_loaded", False):
        try:
            rag.load(base_dir)
        except TypeError:          # 인자를 받지 않는 load()
            rag.load()
        rag._app_loaded = True
    _loaded["ok"] = True


def _prepare_local_images(base_dir):
    """후보 카드용 DB 사진을 로컬에 풀어둠 (외부 URL이 안 열리는 경우 대비)"""
    if os.path.isdir(LOCAL_IMG_DIR) and os.listdir(LOCAL_IMG_DIR):
        return
    zp = os.path.join(base_dir, "images", "pill_images.zip")
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as z:
            z.extractall(LOCAL_IMG_DIR)
        print(f"[app] 후보 사진 {len(os.listdir(LOCAL_IMG_DIR))}장 준비")
    else:
        print(f"[app] {zp} 없음 → 후보 사진은 이미지 URL로 표시")


def _cand_image(c):
    local = os.path.join(LOCAL_IMG_DIR, f"{c['item_seq']}.jpg")
    if os.path.exists(local):
        return local
    url = c.get("image_url") or ""
    return url if url.startswith("http") else None


# ------------------------------------------------------------------ 도우미
def _name(item_seq):
    return identify.S["name_of"].get(str(item_seq), str(item_seq))


def _format_sources(res):
    """RAG sources → '출처: 약이름(섹션), ...' 한 줄"""
    seen, parts = set(), []
    for s in (res.get("sources") or []):
        if isinstance(s, dict):
            code = s.get("item_seq") or s.get("ITEM_SEQ") or s.get("itemSeq")
            label = s.get("name") or (_name(code) if code else "")
            sec = s.get("section") or ""
            key = (label, sec)
            if label and key not in seen:
                seen.add(key)
                parts.append(f"{label}({sec})" if sec else label)
        elif s:
            parts.append(str(s))
    return ("\n\n📚 출처: 식약처 e약은요 · " + ", ".join(parts[:5])) if parts else ""


def _ask_rag(question, item_seq=None):
    try:
        if item_seq and USE_TEMPLATE_MODE and question is None:
            res = rag.search(None, item_seq=item_seq)
        else:
            res = rag.search(question or INFO_QUESTION, item_seq=item_seq)
        return str(res.get("answer", "")) + _format_sources(res)
    except Exception as e:
        traceback.print_exc()
        return f"답변을 만드는 중 오류가 났어요: {e}"


def _is_low_confidence(debug):
    """1등 후보에 각인이나 제품명 근거가 없으면 확신 낮음 (생김새만으로 고른 경우)"""
    rows = debug.get("parts") or []
    if not rows:
        return True
    _, _, parts = rows[0]
    return not (parts.get("name") or parts.get("imprint", 0) >= IMPRINT_EVIDENCE)


def _new_chatbot():
    try:
        return gr.Chatbot(type="messages", height=520, label="대화")
    except TypeError:               # Gradio 버전에 따라 type 인자가 없을 수 있음
        return gr.Chatbot(height=520, label="대화")


# ------------------------------------------------------------------ 이벤트
def on_identify(file_path):
    """사진 → 후보 3개. 진행 상황을 단계별로 보여줌."""
    if not file_path:
        yield (None, [], gr.update(choices=[], value=None), "사진을 먼저 올려주세요.", [])
        return

    try:
        img = identify.load_photo(file_path)
    except Exception as e:
        yield (None, [], gr.update(choices=[], value=None), f"사진을 열 수 없어요: {e}", [])
        return

    yield (img, [], gr.update(choices=[], value=None), "🔍 약을 찾는 중이에요… (10초 정도 걸려요)", [])

    try:
        cands, debug = identify.identify(img, return_debug=True)
    except Exception as e:
        traceback.print_exc()
        yield (img, [], gr.update(choices=[], value=None), f"식별 중 오류가 났어요: {e}", [])
        return

    if not cands:
        yield (img, [], gr.update(choices=[], value=None), "일치하는 약을 찾지 못했어요.", [])
        return

    gallery = [(_cand_image(c), f"{i + 1}. {c['name']}") for i, c in enumerate(cands) if _cand_image(c)]
    choices = [(f"{i + 1}. {c['name']}", c["item_seq"]) for i, c in enumerate(cands)]

    if _is_low_confidence(debug):
        msg = ("🤔 **정확히 찾지 못했어요.** 생김새가 비슷한 약 후보예요.\n\n"
               "포장의 **제품명**이나 알약의 **각인 글자**가 잘 보이게 다시 찍으면 더 정확해져요. "
               "아래 후보 중 맞는 약이 있으면 선택하세요.")
    else:
        msg = "✅ 후보를 찾았어요. 실물과 같은 약을 **직접 확인하고 선택**한 뒤 '약 정보 보기'를 눌러주세요."

    yield (img, gallery, gr.update(choices=choices, value=None), msg, cands)


def on_show_info(selected, history):
    history = list(history or [])
    if not selected:
        history.append({"role": "assistant", "content": "먼저 후보 중에서 약을 선택해 주세요."})
        yield history, None
        return
    name = _name(selected)
    history.append({"role": "user", "content": f"📷 선택한 약: **{name}**"})
    history.append({"role": "assistant", "content": "📖 약 정보를 정리하는 중이에요…"})
    yield history, selected

    history[-1] = {"role": "assistant", "content": _ask_rag(None if USE_TEMPLATE_MODE else INFO_QUESTION, selected)}
    yield history, selected


def on_chat(message, history, current_item, about_current):
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        yield history, ""
        return
    item = current_item if (about_current and current_item) else None
    prefix = f"(💊 {_name(item)}) " if item else ""
    history.append({"role": "user", "content": prefix + message})
    history.append({"role": "assistant", "content": "💬 답변을 만드는 중이에요…"})
    yield history, ""

    history[-1] = {"role": "assistant", "content": _ask_rag(message, item)}
    yield history, ""


def on_reset():
    return (None, None, [], gr.update(choices=[], value=None), "", [], [], None)


# ------------------------------------------------------------------ 화면
def build():
    try:
        blocks = gr.Blocks(title="알약 정보 챗봇", theme=THEME, css=CSS)
    except TypeError:                 # 테마·CSS를 launch()에서 받는 Gradio 버전
        blocks = gr.Blocks(title="알약 정보 챗봇")
    with blocks as demo:
        gr.Markdown("# 💊 알약 정보 챗봇\n알약 사진으로 약을 찾거나, 증상을 입력해 관련 일반의약품 정보를 확인하세요.",
                    elem_id="title")
        gr.Markdown(DISCLAIMER, elem_id="disclaimer")

        cands_state = gr.State([])
        current_item = gr.State(None)

        with gr.Row():
            # ---------------- 왼쪽: 사진으로 찾기
            with gr.Column(scale=4):
                gr.Markdown("### ① 사진으로 약 찾기")
                upload = gr.File(label="알약 사진 (알약, 은박 포장, 상자 모두 가능)",
                                 file_types=["image", ".heic", ".HEIC"], type="filepath")
                preview = gr.Image(label="올린 사진", interactive=False, height=220)
                find_btn = gr.Button("🔍 약 찾기", variant="primary", elem_id="find-btn")
                status = gr.Markdown(elem_id="status")
                gallery = gr.Gallery(label="후보 (DB 사진)", columns=3, height=200, allow_preview=True)
                choice = gr.Radio(label="실물과 같은 약을 선택하세요", choices=[])
                info_btn = gr.Button("📖 약 정보 보기", elem_id="info-btn")
                reset_btn = gr.Button("처음부터 다시", variant="secondary")

            # ---------------- 오른쪽: 대화
            with gr.Column(scale=6):
                gr.Markdown("### ② 질문하기")
                chatbot = _new_chatbot()
                about_current = gr.Checkbox(label="선택한 약에 대해 질문하기 (끄면 전체 약에서 검색)", value=True)
                with gr.Row():
                    msg = gr.Textbox(placeholder="예) 머리가 아프고 열이 나요 / 공복에 먹어도 돼요?",
                                     show_label=False, scale=8)
                    send = gr.Button("보내기", variant="primary", scale=1)
                gr.Examples(["머리가 아프고 열이 나요", "콧물이 나고 재채기가 계속 나와요",
                             "소화가 안 되고 속이 더부룩해요", "이 약 공복에 먹어도 돼요?"], inputs=msg)

        find_btn.click(on_identify, inputs=upload,
                       outputs=[preview, gallery, choice, status, cands_state])
        info_btn.click(on_show_info, inputs=[choice, chatbot], outputs=[chatbot, current_item])
        send.click(on_chat, inputs=[msg, chatbot, current_item, about_current], outputs=[chatbot, msg])
        msg.submit(on_chat, inputs=[msg, chatbot, current_item, about_current], outputs=[chatbot, msg])
        reset_btn.click(on_reset, outputs=[upload, preview, gallery, choice, status, cands_state, chatbot, current_item])
    return demo


def launch(base_dir, share=True, **kwargs):
    load(base_dir)
    gr.close_all()                    # 이전에 띄운 앱이 있으면 닫기
    demo = build()
    demo.queue()
    try:
        demo.launch(share=share, theme=THEME, css=CSS, **kwargs)
    except TypeError:                 # 테마·CSS를 Blocks()에서 받는 Gradio 버전
        demo.launch(share=share, **kwargs)
    return demo


if __name__ == "__main__":
    launch(os.environ.get("PILL_BASE", "."), share=True)
