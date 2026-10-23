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
import base64

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BG_FILE = "bg.jpg"          # assets 폴더 안의 배경 이미지 파일명
ICON_FILE = "icon1.png"      # 제목 왼쪽 아이콘 (배경이 투명한 PNG 추천)
ICON2_FILE = "icon2.png"    # 예시 질문 오른쪽 아이콘 ("" 이면 안 나옴)
ICON2_SIZE = "90px"         # 그 아이콘 높이
ICON_SIZE = "90px"          # 아이콘 높이

# 배경 그림에서 뽑은 색
C_SKY = "#C8E8E7"       # 하늘색 배경
C_BROWN = "#5A4636"     # 갈색 테두리 선
C_CREAM = "#FFF8E7"     # 크림색 카드
C_YELLOW = "#F7D84A"    # 노란 병아리
C_PINK = "#F4C3CB"      # 분홍 귀
C_TEXT = "#4A3A2C"      # 글씨

# 배경 그림은 화면을 꽉 채우지 않고 왼쪽 아래에 이 높이로만 붙여요 (너무 크면 화질이 깨져 보여서)
BG_HEIGHT = "min(80vh, 760px)"
CHAT_HEIGHT = "calc(100vh - 480px)"   # 대화창 높이: 화면 높이에 맞춰 한 화면에 들어오게


def _img_data_uri(filename):
    path = os.path.join(ASSET_DIR, filename)
    if not os.path.exists(path):
        return ""
    ext = filename.rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")
    with open(path, "rb") as f:
        return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode()


def _make_theme():
    base = gr.themes.Soft(primary_hue="yellow", secondary_hue="stone", neutral_hue="stone",
                          font=[gr.themes.GoogleFont("Gowun Dodum"), "sans-serif"])
    try:
        return base.set(
            body_text_color=C_TEXT,
            block_radius="18px",
            input_radius="14px",
            button_large_radius="999px",
            button_primary_background_fill=C_YELLOW,
            button_primary_background_fill_hover="#F2CC2E",
            button_primary_text_color=C_BROWN,
            button_secondary_background_fill=C_CREAM,
            button_secondary_text_color=C_BROWN,
        )
    except Exception:          # Gradio 버전에 따라 일부 항목 이름이 다를 수 있음
        return base


THEME = _make_theme()
BG = _img_data_uri(BG_FILE)
ICON = _img_data_uri(ICON_FILE)
ICON2 = _img_data_uri(ICON2_FILE) if ICON2_FILE else ""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Jua&family=Gowun+Dodum&display=swap');

/* ---------- 페이지 배경 ---------- */
:root { --bgh: %(bgh)s; }
body, gradio-app { background-color: %(sky)s !important; }
%(bg_rule)s
.gradio-container {
    background: transparent !important;
    max-width: none !important;
    /* 캐릭터 너비(배경 높이의 약 0.45배)만큼 오른쪽으로 밀기 */
    padding: 14px 2.5vw 14px calc(var(--bgh) * 0.47 + 24px) !important;
}
.gradio-container > .main, .gradio-container .contain { gap: 12px !important; }
footer { display: none !important; }

/* ---------- 스티커 카드 (갈색 테두리 + 밀린 그림자) ---------- */
.card {
    background: %(cream)s !important;
    border: 2.5px solid %(brown)s !important;
    border-radius: 24px !important;
    box-shadow: 5px 5px 0 %(brown)s !important;
    padding: 12px 18px !important;
}
.card .block, .card .form { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ---------- 제목 카드 (작게) ---------- */
#hero-card { padding: 10px 18px !important; }
.hero { display: flex; align-items: center; gap: 14px; }
.hero-icon { height: %(icon)s; width: auto; flex-shrink: 0; }
.hero-text { flex: 1; min-width: 0; }
.hero-icon-right { margin-left: auto; }
.hero-text h1 { font-family: 'Jua', sans-serif !important; font-size: 1.7em !important; color: %(brown)s !important;
                margin: 0 !important; line-height: 1.2; }
.hero-text .sub { margin: 2px 0 6px 0 !important; color: %(text)s; font-size: 0.95em; }
.hero-text .disclaimer {
    display: inline-block; background: #FFFFFF; border: 2px dashed %(pink)s; border-radius: 12px;
    padding: 3px 10px; font-size: 0.76em; line-height: 1.5; color: %(text)s;
}
.section-title h3 { font-family: 'Jua', sans-serif !important; font-size: 1.25em !important; color: %(brown)s !important; margin: 0 !important; }

/* ---------- 버튼 ---------- */
button.primary, button.secondary { border: 2.5px solid %(brown)s !important; font-family: 'Jua', sans-serif !important; font-size: 1.1em !important; }
button.primary { box-shadow: 3px 3px 0 %(brown)s !important; }
button.primary:active { transform: translate(2px, 2px); box-shadow: 1px 1px 0 %(brown)s !important; }
#reset-btn { font-size: 0.9em !important; opacity: 0.85; }

/* ---------- 입력칸 ---------- */
.card textarea, .card input[type=text] { background: #FFFFFF !important; border: 2px solid %(brown)s !important; }
/* 질문 입력창: 테두리를 바깥 틀에 그려서 모서리가 잘리지 않게 */
#send-row { align-items: center !important; gap: 12px !important; padding: 2px 6px 6px 2px; overflow: visible !important; }
#msg { background: #FFFFFF !important; border: 2px solid %(brown)s !important; border-radius: 999px !important;
       box-shadow: 3px 3px 0 %(brown)s !important; padding: 0 14px !important; overflow: hidden; }
#example-row { align-items: center !important; }
.side-icon { height: %(icon2)s !important; width: auto !important; max-width: 100%% !important; object-fit: contain; display: block; margin: 0 auto; }
#msg textarea, #msg input { border: none !important; box-shadow: none !important; background: transparent !important;
    padding: 8px 0 !important; min-height: 0 !important; height: 40px !important; resize: none; }
#upload { border: 2.5px dashed %(brown)s !important; border-radius: 18px !important; background: #FFFFFF !important; }

/* ---------- 상태 메시지 ---------- */
#status p { background: #FFFFFF; border-radius: 14px; padding: 10px 14px; border-left: 6px solid %(yellow)s; }

/* ---------- 대화창 말풍선 ---------- */
#chat { background: #FFFFFF !important; border: 2px solid %(brown)s !important; border-radius: 18px !important; min-height: 260px; }
/* 예시 질문은 작게 */
.card .examples button, .card .gallery-item { font-size: 0.82em !important; padding: 4px 10px !important; }
#chat .message.user, #chat .user-row .message { background: %(pink)s !important; border-color: %(brown)s !important; }
#chat .message.bot, #chat .bot-row .message { background: %(cream)s !important; border-color: %(brown)s !important; }

/* ---------- 화면이 좁을 때 (노트북 작은 화면, 휴대폰) ---------- */
@media (max-width: 1000px) {
    .gradio-container { padding: 16px !important; }
    body, gradio-app { background-image: none !important; }
}
""" % {
    "sky": C_SKY, "brown": C_BROWN, "cream": C_CREAM, "yellow": C_YELLOW, "pink": C_PINK, "text": C_TEXT,
    "bgh": BG_HEIGHT, "icon": ICON_SIZE, "icon2": ICON2_SIZE,
    "bg_rule": (("body, gradio-app { background: "
                 "url('" + BG + "') left bottom / auto var(--bgh) no-repeat fixed, "   # 그림: 왼쪽 아래, 정해진 높이
                 "linear-gradient(" + C_BROWN + ", " + C_BROWN + ") left top / calc(var(--bgh) * 0.0125) 100% no-repeat fixed, "  # 왼쪽 벽 선을 위까지 이어 그림
                 + C_SKY + " !important; }")
                if BG else ""),
}

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
        return gr.Chatbot(type="messages", height=CHAT_HEIGHT, show_label=False, elem_id="chat")
    except TypeError:               # Gradio 버전에 따라 type 인자가 없을 수 있음
        return gr.Chatbot(height=CHAT_HEIGHT, show_label=False, elem_id="chat")


# ------------------------------------------------------------------ 이벤트
def on_upload(file_path):
    """사진을 올리면 바로 미리보기 (HEIC도 변환해서 보여줌)"""
    if not file_path:
        return gr.update(value=None, visible=False)
    try:
        return gr.update(value=identify.load_photo(file_path), visible=True)
    except Exception:
        return gr.update(value=None, visible=False)


def on_identify(file_path):
    """사진 → 후보 3개. 진행 상황을 단계별로 보여줌.
    출력: preview, gallery, choice, status, cands_state, cand_group"""
    hide = gr.update(visible=False)
    empty_choice = gr.update(choices=[], value=None)
    if not file_path:
        yield (gr.update(), [], empty_choice, "사진을 먼저 올려주세요 📷", [], hide)
        return

    try:
        img = identify.load_photo(file_path)
    except Exception as e:
        yield (gr.update(), [], empty_choice, f"사진을 열 수 없어요: {e}", [], hide)
        return

    yield (gr.update(value=img, visible=True), [], empty_choice,
           "🔍 약을 찾는 중이에요… (10초 정도 걸려요)", [], hide)

    try:
        cands, debug = identify.identify(img, return_debug=True)
    except Exception as e:
        traceback.print_exc()
        yield (gr.update(), [], empty_choice, f"식별 중 오류가 났어요: {e}", [], hide)
        return

    if not cands:
        yield (gr.update(), [], empty_choice, "일치하는 약을 찾지 못했어요.", [], hide)
        return

    gallery = [(_cand_image(c), f"{i + 1}. {c['name']}") for i, c in enumerate(cands) if _cand_image(c)]
    choices = [(f"{i + 1}. {c['name']}", c["item_seq"]) for i, c in enumerate(cands)]

    if _is_low_confidence(debug):
        msg = ("🤔 **정확히 찾지 못했어요.** 생김새가 비슷한 약 후보예요.\n\n"
               "포장의 **제품명**이나 알약의 **각인 글자**가 잘 보이게 다시 찍으면 더 정확해져요.")
    else:
        msg = "✅ 후보를 찾았어요! 실물과 같은 약을 **직접 확인하고** 골라주세요."

    yield (gr.update(), gallery, gr.update(choices=choices, value=None), msg, cands, gr.update(visible=True))


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
    # upload, preview, gallery, choice, status, cands_state, chatbot, current_item, cand_group
    return (None, gr.update(value=None, visible=False), [], gr.update(choices=[], value=None), "",
            [], [], None, gr.update(visible=False))


# ------------------------------------------------------------------ 화면
def build():
    try:
        blocks = gr.Blocks(title="알약 정보 챗봇", theme=THEME, css=CSS)
    except TypeError:                 # 테마·CSS를 launch()에서 받는 Gradio 버전
        blocks = gr.Blocks(title="알약 정보 챗봇")
    with blocks as demo:
        cands_state = gr.State([])
        current_item = gr.State(None)

        # ---------------- 상단 제목 카드 (아이콘 + 제목 + 면책 문구)
        with gr.Column(elem_classes="card", elem_id="hero-card"):
            icon_html = f'<img class="hero-icon" src="{ICON}" alt="">' if ICON else '<span style="font-size:2.4em">💊</span>'
            disclaimer_html = DISCLAIMER.replace("**", "<b>", 1).replace("**", "</b>", 1)
            gr.HTML(f"""
            <div class="hero">
              {icon_html}
              <div class="hero-text">
                <h1>알약 정보 챗봇</h1>
                <p class="sub">알약 사진으로 무슨 약인지 찾고, 궁금한 점을 물어보세요.</p>
                <span class="disclaimer">{disclaimer_html}</span>
              </div>
            </div>""")

        with gr.Row(equal_height=False):
            # ---------------- 왼쪽 카드: 사진으로 찾기
            with gr.Column(scale=5, elem_classes="card"):
                gr.Markdown("### ① 사진으로 약 찾기", elem_classes="section-title")
                upload = gr.File(label="알약 · 은박 포장 · 상자 사진", file_types=["image", ".heic", ".HEIC"],
                                 type="filepath", height=120, elem_id="upload")
                preview = gr.Image(show_label=False, interactive=False, height=220, visible=False)
                find_btn = gr.Button("🔍 약 찾기", variant="primary", size="lg", elem_id="find-btn")
                status = gr.Markdown(elem_id="status")
                with gr.Column(visible=False) as cand_group:
                    gallery = gr.Gallery(show_label=False, columns=3, height=190, allow_preview=True,
                                         object_fit="contain")
                    choice = gr.Radio(label="실물과 같은 약을 골라주세요", choices=[])
                    info_btn = gr.Button("📖 이 약 정보 보기", variant="primary", elem_id="info-btn")
                reset_btn = gr.Button("↺ 처음부터 다시", variant="secondary", size="sm", elem_id="reset-btn")

            # ---------------- 오른쪽 카드: 대화
            with gr.Column(scale=6, elem_classes="card"):
                gr.Markdown("### ② 궁금한 점 물어보기", elem_classes="section-title")
                chatbot = _new_chatbot()
                about_current = gr.Checkbox(label="고른 약에 대해 질문하기 (끄면 전체 약에서 찾아요)", value=True)
                with gr.Row(elem_id="send-row"):
                    msg = gr.Textbox(placeholder="예) 머리가 아프고 열이 나요 / 공복에 먹어도 돼요?",
                                     lines=1, max_lines=1, show_label=False, scale=8, container=False, elem_id="msg")
                    send = gr.Button("보내기", variant="primary", scale=1, min_width=90)
                with gr.Row(elem_id="example-row", equal_height=False):
                    with gr.Column(scale=8, min_width=200):
                        gr.Examples(["머리가 아프고 열이 나요", "콧물이 나고 재채기가 계속 나와요",
                                     "소화가 안 되고 속이 더부룩해요", "이 약 공복에 먹어도 돼요?"],
                                    inputs=msg, label="이렇게 물어볼 수 있어요")
                    if ICON2:
                        with gr.Column(scale=1, min_width=90):
                            gr.HTML(f'<img class="side-icon" src="{ICON2}" alt="">')

        upload.change(on_upload, inputs=upload, outputs=preview)
        find_btn.click(on_identify, inputs=upload,
                       outputs=[preview, gallery, choice, status, cands_state, cand_group])
        info_btn.click(on_show_info, inputs=[choice, chatbot], outputs=[chatbot, current_item])
        send.click(on_chat, inputs=[msg, chatbot, current_item, about_current], outputs=[chatbot, msg])
        msg.submit(on_chat, inputs=[msg, chatbot, current_item, about_current], outputs=[chatbot, msg])
        reset_btn.click(on_reset, outputs=[upload, preview, gallery, choice, status, cands_state,
                                           chatbot, current_item, cand_group])
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
