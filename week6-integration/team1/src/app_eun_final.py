"""
알약 정보 챗봇 - 대체 디자인 (탭 구조)

app.py 의 처리 함수(식별, 답변)를 그대로 쓰고 화면만 다르게 만든 버전이에요.
    탭 1 "사진으로 약 찾기": 사진 → 후보 선택 → 약 정보 → 그 약에 대해 질문
    탭 2 "증상으로 약 찾기": 자주 찾는 증상 버튼 + 자유 입력

코랩에서 실행 (app.py, identify.py, rag.py 와 같은 폴더)
    import sys
    sys.path.append(PROJECT)
    import app_alt
    app_alt.launch(BASE, share=True)
"""

import os

import gradio as gr

import app as base

# ------------------------------------------------------------------ 디자인 (app.py 와 같은 배경·색감)
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(HERE, "assets")   # 기본: app_eun.py 옆의 assets 폴더
BG_FILE = "bg.jpg"          # assets 폴더 안의 배경 그림
ICON_FILE = "icon1.png"     # 제목 왼쪽 아이콘 (없으면 💊)
ICON_SIZE = "68px"          # 아이콘 높이
ICON2_FILE = ""    # 제목 카드 오른쪽 아래 아이콘 ("" 이면 안 나옴)
ICON2_SIZE = "80px"         # 그 아이콘 높이
CHIP_COLS_WIDE = 6          # 증상 버튼: 넓은 화면에서 한 줄에 몇 개
CHIP_COLS_NARROW = 3        # 증상 버튼: 좁은 화면에서 한 줄에 몇 개 (6개면 3×2)
BG_HEIGHT = "min(80vh, 760px)"   # 배경 그림 높이 (왼쪽 아래에 붙음)

# 배경 그림에서 뽑은 색
C_SKY = "#C8E8E7"       # 하늘색 배경
C_BROWN = "#5A4636"     # 갈색 테두리 선
C_CREAM = "#FFF8E7"     # 크림색 카드
C_YELLOW = "#F7D84A"    # 노란 병아리
C_PINK = "#F4C3CB"      # 분홍 귀
C_TEXT = "#4A3A2C"      # 글씨


def _find_asset(filename):
    """assets 폴더에서 파일을 찾는다 (대소문자 무시: bg.JPG, Icon1.PNG 도 찾음)"""
    if not filename:
        return None
    for d in [ASSET_DIR, os.path.join(HERE, "assets"), os.path.join(os.getcwd(), "assets")]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower() == filename.lower() and os.path.isfile(os.path.join(d, f)):
                return os.path.join(d, f)
    return None


def _img_data_uri(filename):
    path = _find_asset(filename)
    if not path:
        return ""
    ext = path.rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")
    with open(path, "rb") as f:
        return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode()


def _make_theme():
    t = gr.themes.Soft(primary_hue="yellow", secondary_hue="stone", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Gowun Dodum"), "sans-serif"])
    try:
        return t.set(
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
        return t


THEME = _make_theme()
CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Jua&family=Gowun+Dodum&display=swap');

/* ---------- 페이지 배경 ---------- */
:root { --bgh: %(bgh)s; }
body, gradio-app { background-color: %(sky)s !important; }
%(bg_rule)s
.gradio-container {
    background: transparent !important;
    max-width: none !important;
    /* 왼쪽 아래 캐릭터를 가리지 않게 오른쪽으로 밀기 */
    padding: 14px 2.5vw 14px calc(var(--bgh) * 0.47 + 24px) !important;
}
footer { display: none !important; }

/* ---------- 제목 카드 ---------- */
.hero {
    display: flex; align-items: center; gap: 14px;
    background: %(cream)s; border: 2.5px solid %(brown)s; border-radius: 24px;
    box-shadow: 5px 5px 0 %(brown)s; padding: 12px 22px; margin-bottom: 8px;
    position: relative;
    display: flex; align-items: center; gap: 14px;
}
.hero-icon { height: %(icon)s; width: auto; flex-shrink: 0; }
.hero-text { flex: 1; min-width: 0; }
.hero-icon2 { position: absolute; right: 40px; top: calc(100%% + 6px); height: %(icon2)s !important; width: auto !important;
              object-fit: contain; z-index: 5; pointer-events: none; }
.hero h1 { font-family: 'Jua', sans-serif !important; color: %(brown)s !important; margin: 0 0 2px 0 !important;
           font-size: 1.8em !important; line-height: 1.2; }
.hero p { color: %(text)s !important; margin: 0 !important; }

/* ---------- 탭 ---------- */
.tabs button[role=tab], .tab-nav button {
    font-family: 'Jua', sans-serif !important; font-size: 1.1em !important; color: %(brown)s !important;
    background: transparent !important; border: none !important; opacity: 0.7;
}
.tabs button[role=tab][aria-selected=true], .tab-nav button.selected {
    opacity: 1; background: %(yellow)s !important; border: 2.5px solid %(brown)s !important;
    border-radius: 999px !important; box-shadow: 3px 3px 0 %(brown)s !important;
}
.tabs, .tabitem { background: transparent !important; border: none !important; }
.tab-nav, .tab-wrapper, .tab-container, [role=tablist] { border-bottom: none !important; box-shadow: none !important; }
.tab-wrapper::after, .tab-container::after, .tab-nav::after { display: none !important; }

/* ---------- 스티커 카드 ---------- */
.card {
    background: %(cream)s !important; border: 2.5px solid %(brown)s !important;
    border-radius: 24px !important; box-shadow: 5px 5px 0 %(brown)s !important; padding: 14px 18px !important;
}
.card .block, .card .form { background: transparent !important; border: none !important; box-shadow: none !important; }
.step {
    display: inline-block; width: 1.7em; height: 1.7em; line-height: 1.55em; text-align: center;
    border-radius: 50%%; background: %(yellow)s; color: %(brown)s; border: 2px solid %(brown)s;
    font-family: 'Jua', sans-serif; font-weight: 700; margin-right: 8px;
}
.card strong, .step-title strong { font-family: 'Jua', sans-serif; color: %(brown)s; font-size: 1.15em; font-weight: 400; }

/* ---------- 입력칸 / 사진칸 ---------- */
.card textarea, .card input[type=text], .card .image-container, .card .upload-container {
    background: #FFFFFF !important;
}
textarea { border: 2px solid %(brown)s !important; border-radius: 14px !important; background: #FFFFFF !important; }
#picked { min-height: 1.6em; font-weight: 600; color: %(brown)s; }

/* ---------- 버튼 ---------- */
button.primary, button.secondary { border: 2.5px solid %(brown)s !important; font-family: 'Jua', sans-serif !important; }
button.primary { box-shadow: 3px 3px 0 %(brown)s !important; }
button.primary:active { transform: translate(2px, 2px); box-shadow: 1px 1px 0 %(brown)s !important; }
#find-btn, #info-btn, #send1, #send2 { font-size: 1.1em !important; }
.chip button, button.chip {
    border-radius: 999px !important; background: #FFFFFF !important; border: 2px solid %(brown)s !important;
    color: %(brown)s !important; font-family: 'Gowun Dodum', sans-serif !important;
}
.chip button:hover, button.chip:hover { background: %(pink)s !important; }
#ask-current, #ask-current * { white-space: nowrap; }
/* 체크박스 문구가 잘리지 않게: 글자 길이만큼 자리 확보, 입력칸이 대신 줄어듦 */
#ask-current { flex: 0 0 auto !important; min-width: max-content !important; width: auto !important; }
#msg1 { min-width: 0 !important; }

/* 증상 버튼: 격자로 고정 (넓으면 6개 한 줄, 좁으면 3개씩 2줄) */
#chips { display: grid !important; grid-template-columns: repeat(%(chip_wide)s, minmax(0, 1fr)) !important; gap: 10px !important; }
#chips > * { min-width: 0 !important; width: 100%% !important; }
@media (max-width: 1250px) { #chips { grid-template-columns: repeat(%(chip_narrow)s, minmax(0, 1fr)) !important; } }

/* ---------- 대화창 ---------- */
.gradio-container .chatbot, .gradio-container [data-testid=chatbot] {
    background: #FFFFFF !important; border: 2.5px solid %(brown)s !important; border-radius: 18px !important;
}
/* 말풍선: 글자 길이에 맞게, 여백 작게 */
.message.user, .user-row .message { background: %(pink)s !important; border: 2px solid %(brown)s !important; }
.message.bot, .bot-row .message { background: %(cream)s !important; border: 2px solid %(brown)s !important; }
.message.user, .message.bot, .user-row .message, .bot-row .message {
    width: fit-content !important; max-width: 80%% !important; min-height: 0 !important;
    padding: 8px 14px !important; border-radius: 16px !important;
}
/* 말풍선 안쪽 요소에 테두리·여백이 또 생기지 않게 */
.message .message, .message > div, .message .prose, .message button {
    border: none !important; background: transparent !important; box-shadow: none !important;
    padding: 0 !important; margin: 0 !important; min-height: 0 !important;
}
.message p { margin: 0 !important; }

/* ---------- 아래 안내문 ---------- */
.footer {
    font-size: 0.86em; color: %(text)s; background: #FFFFFF; border: 2px dashed %(pink)s; border-radius: 14px;
    padding: 8px 14px; margin-top: 12px; word-break: keep-all; text-wrap: balance;
}

/* ---------- 화면이 좁을 때 ---------- */
@media (max-width: 1000px) {
    .gradio-container { padding: 16px !important; }
    body, gradio-app { background-image: none !important; }
}
"""

CSS, HERO, BG, ICON, ICON2 = "", "", "", "", ""


def _make_design():
    """배경·아이콘을 읽어서 CSS 와 제목 HTML 을 만든다. launch() 때마다 다시 실행됨."""
    global CSS, HERO, BG, ICON, ICON2
    BG = _img_data_uri(BG_FILE)
    ICON = _img_data_uri(ICON_FILE)
    ICON2 = _img_data_uri(ICON2_FILE) if ICON2_FILE else ""
    CSS = CSS_TEMPLATE % {
        "sky": C_SKY, "brown": C_BROWN, "cream": C_CREAM, "yellow": C_YELLOW, "pink": C_PINK, "text": C_TEXT,
        "bgh": BG_HEIGHT, "icon": ICON_SIZE, "icon2": ICON2_SIZE,
        "chip_wide": CHIP_COLS_WIDE, "chip_narrow": CHIP_COLS_NARROW,
        "bg_rule": (("body, gradio-app { background: "
                     "url('" + BG + "') left bottom / auto var(--bgh) no-repeat fixed, "
                     "linear-gradient(" + C_BROWN + ", " + C_BROWN + ") left top / calc(var(--bgh) * 0.0125) 100% no-repeat fixed, "
                     + C_SKY + " !important; }") if BG else ""),
    }
    icon_html = f'<img class="hero-icon" src="{ICON}" alt="">' if ICON else '<span style="font-size:2.2em">💊</span>'
    icon2_html = f'<img class="hero-icon2" src="{ICON2}" alt="">' if ICON2 else ""
    HERO = f"""
<div class="hero">
  {icon_html}
  <div class="hero-text">
    <h1>알약 정보 도우미</h1>
    <p>사진으로 약을 확인하거나, 증상을 말하면 관련 일반의약품의 식약처 공식 정보를 알려드려요.</p>
  </div>
  {icon2_html}
</div>
"""
    print(f"[디자인] assets: {ASSET_DIR}")
    print(f"[디자인] 배경 {BG_FILE}: {'OK' if BG else '못 찾음'} / 아이콘 {ICON_FILE}: {'OK' if ICON else '못 찾음 (💊 사용)'}"
          f" / 아이콘2 {ICON2_FILE or '(끔)'}: {'OK' if ICON2 else '없음'}")


_make_design()

FOOTER = ("<div class='footer'>⚠️ 식품의약품안전처 공개 자료를 바탕으로 한 <b>참고용</b> 정보이며 진단이나 처방이 아닙니다.<br>"
          "복용 전 약사·의사와 상담하세요. 낱알 식별이 가능한 국내 일반의약품(정제·캡슐 등)만 지원합니다.</div>")

SYMPTOM_CHIPS = [
    ("🤕 두통·발열", "머리가 아프고 열이 나요"),
    ("🤧 콧물·재채기", "콧물이 나고 재채기가 계속 나와요"),
    ("🤢 소화불량", "소화가 안 되고 속이 더부룩해요"),
    ("😷 기침·가래", "기침이 나고 가래가 끓어요"),
    ("🔥 속쓰림", "속이 쓰려요"),
    ("🚑 설사", "설사를 해요"),
]


def _chatbot(label, height):
    try:
        return gr.Chatbot(type="messages", height=height, label=label)
    except TypeError:               # Gradio 버전에 따라 type 인자가 없을 수 있음
        return gr.Chatbot(height=height, label=label)


def _set_text(text):
    return lambda: text


def _picked_label(selected):
    return f"✅ 선택한 약: **{base._name(selected)}**" if selected else ""


def _format_sources(res):
    """출처를 약 하나에 섹션을 묶어서 한 줄로: 약이름 [복용법·경고·주의사항]"""
    by = {}
    for s in (res.get("sources") or []):
        if isinstance(s, dict) and s.get("name"):
            secs = by.setdefault(s["name"], [])
            if s.get("section") and s["section"] not in secs:
                secs.append(s["section"])
    parts = [f"{n} [{'·'.join(secs)}]" if secs else n for n, secs in by.items()]
    return ("\n\n📚 출처: 식약처 e약은요 · " + " / ".join(parts[:5])) if parts else ""


def build():
    try:
        blocks = gr.Blocks(title="알약 정보 도우미", theme=THEME, css=CSS)
    except TypeError:               # 테마·CSS를 launch()에서 받는 Gradio 버전
        blocks = gr.Blocks(title="알약 정보 도우미")

    with blocks as demo:
        gr.HTML(HERO)

        cands_state = gr.State([])
        current_item = gr.State(None)
        no_item = gr.State(None)       # 증상 탭: 특정 약 없이 질문
        ask_all = gr.State(False)

        with gr.Tabs():
            # ======================================================= 탭 1: 사진으로
            with gr.Tab("📷 사진으로 약 찾기"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="card"):
                        gr.Markdown("<span class='step'>1</span> **사진 올리기**")
                        upload = gr.File(label="알약, 은박 포장, 상자 모두 가능",
                                         file_types=["image", ".heic", ".HEIC"], type="filepath")
                        preview = gr.Image(label="올린 사진", interactive=False, height=200)
                        find_btn = gr.Button("🔍 약 찾기", variant="primary", elem_id="find-btn")
                        status = gr.Markdown()
                        reset_btn = gr.Button("처음부터 다시", variant="secondary", size="sm")

                    with gr.Column(scale=7, elem_classes="card"):
                        gr.Markdown("<span class='step'>2</span> **후보에서 같은 약 고르기**")
                        gallery = gr.Gallery(label="후보 (DB 사진)", columns=3, height=190, allow_preview=True)
                        gr.Markdown("<span style='font-size:0.9em;opacity:0.8'>💡 후보 사진을 누르면 크게 볼 수 있어요. "
                                    "실물과 비교해 보고 아래에서 같은 약을 선택하세요.</span>")
                        choice = gr.Radio(label="실물과 같은 약을 선택하세요", choices=[])
                        picked = gr.Markdown(elem_id="picked")
                        info_btn = gr.Button("📖 약 정보 보기", variant="primary", elem_id="info-btn")

                gr.Markdown("<span class='step'>3</span> **약 정보와 질문**")
                chat1 = _chatbot("대화", 380)
                with gr.Row():
                    msg1 = gr.Textbox(placeholder="예) 공복에 먹어도 돼요? / 같이 먹으면 안 되는 약이 있나요?",
                                      show_label=False, scale=7, min_width=200, elem_id="msg1")
                    ask_current = gr.State(True)
                    send1 = gr.Button("보내기", variant="primary", scale=1, elem_id="send1")
                gr.Markdown("<span style='font-size:0.88em;opacity:0.8'>증상으로 약을 찾고 싶으면 '증상으로 약 찾기' 탭을 이용하세요.</span>")

            # ======================================================= 탭 2: 증상으로
            with gr.Tab("💬 증상으로 약 찾기"):
                gr.Markdown("자주 찾는 증상을 누르거나 직접 입력해 보세요.")
                with gr.Row(elem_id="chips"):
                    chip_btns = [gr.Button(label, size="sm", elem_classes="chip", min_width=0) for label, _ in SYMPTOM_CHIPS]
                chat2 = _chatbot("증상 상담", 440)
                with gr.Row():
                    msg2 = gr.Textbox(placeholder="예) 머리가 아프고 열이 나요",
                                      show_label=False, scale=9)
                    send2 = gr.Button("보내기", variant="primary", scale=1, elem_id="send2")

        gr.HTML(FOOTER)

        # ------------------------------------------------------- 이벤트 (처리 함수는 app.py 것을 사용)
        find_btn.click(base.on_identify, inputs=upload,
                       outputs=[preview, gallery, choice, status, cands_state])
        choice.change(_picked_label, inputs=choice, outputs=picked)
        info_btn.click(base.on_show_info, inputs=[choice, chat1], outputs=[chat1, current_item])
        reset_btn.click(base.on_reset,
                        outputs=[upload, preview, gallery, choice, status, cands_state, chat1, current_item])

        ins1 = [msg1, chat1, current_item, ask_current]
        send1.click(base.on_chat, inputs=ins1, outputs=[chat1, msg1])
        msg1.submit(base.on_chat, inputs=ins1, outputs=[chat1, msg1])

        ins2 = [msg2, chat2, no_item, ask_all]
        send2.click(base.on_chat, inputs=ins2, outputs=[chat2, msg2])
        msg2.submit(base.on_chat, inputs=ins2, outputs=[chat2, msg2])
        for btn, (_, text) in zip(chip_btns, SYMPTOM_CHIPS):
            btn.click(_set_text(text), outputs=msg2).then(base.on_chat, inputs=ins2, outputs=[chat2, msg2])
    return demo


def launch(base_dir, share=True, template=True, asset_dir=None, **kwargs):
    """template=True 면 후보 선택 직후 '약 정보 보기'를 LLM 없이 고정 템플릿으로 보여준다 (rag.search(None, item_seq))."""
    global ASSET_DIR
    if asset_dir:                   # 예) app_eun.launch(BASE, asset_dir=f"{MY_DIR}/6주차/assets")
        ASSET_DIR = asset_dir
    _make_design()
    base.USE_TEMPLATE_MODE = template
    base._format_sources = _format_sources
    base.load(base_dir)
    gr.close_all()
    demo = build()
    demo.queue()
    try:
        demo.launch(share=share, theme=THEME, css=CSS, **kwargs)
    except TypeError:               # 테마·CSS를 Blocks()에서 받는 Gradio 버전
        demo.launch(share=share, **kwargs)
    return demo


if __name__ == "__main__":
    launch(os.environ.get("PILL_BASE", "."), share=True)
