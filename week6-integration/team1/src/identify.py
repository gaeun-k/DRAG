"""
알약 사진 → 후보 3개

사용법
    import identify
    identify.load("/content/drive/MyDrive/pill_chatbot")   # 앱 시작할 때 한 번
    identify.identify("photo.jpg")
    # → [{"item_seq", "name", "image_url", "score"}, ...] (score 높은 순 3개)

설정은 {BASE}/index/identify_config.json 에서 읽어요. (04 노트북 E 단계에서 저장)
    weights          : {"imprint", "siglip", "attr"} 가중치
    vlm_model        : "gemini-..." 이면 Gemini API, 그 외(예: "Qwen/Qwen2.5-VL-3B-Instruct")는 로컬 모델
    vlm_fallbacks    : (선택) Gemini가 바쁠 때 넘어갈 예비 모델 목록
    siglip_model     : 인덱스를 만든 SigLIP 모델 (인덱스와 같아야 함)
    name_threshold   : 포장 제품명 매칭 기준 (0~1)
VLM 모델을 바꾸려면 코드는 그대로 두고 이 JSON만 바꾸면 돼요.
Gemini를 쓰면 환경변수 GEMINI_API_KEY 가 필요해요.
"""

import json
import os
import re
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# ---------------------------------------------------------------- 설정 기본값
DEFAULT_CONFIG = {
    "weights": {"imprint": 0.5, "siglip": 0.3, "attr": 0.2},
    "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "vlm_fallbacks": [],
    "siglip_model": "google/siglip2-so400m-patch14-384",
    "name_threshold": 0.85,
}
NAME_BONUS = 1.0
POOL = 50
KEY_COL_CANDIDATES = ["ITEM_SEQ", "item_seq", "itemSeq"]
NAME_COL_CANDIDATES = ["ITEM_NAME", "itemName", "name", "item_name"]
COL_CANDIDATES = {
    "front": ["PRINT_FRONT", "imprint_front", "print_front"],
    "back": ["PRINT_BACK", "imprint_back", "print_back"],
    "color1": ["COLOR_CLASS1", "color1"],
    "color2": ["COLOR_CLASS2", "color2"],
    "shape": ["DRUG_SHAPE", "shape"],
    "image": ["ITEM_IMAGE", "image_url", "itemImage"],
}

COLOR_MAP = {"white": "하양", "yellow": "노랑", "orange": "주황", "pink": "분홍", "red": "빨강",
             "brown": "갈색", "light green": "연두", "yellow green": "연두", "green": "초록",
             "teal": "청록", "cyan": "청록", "blue": "파랑", "navy": "남색", "purple": "보라",
             "violet": "보라", "magenta": "자주", "gray": "회색", "grey": "회색", "black": "검정",
             "transparent": "투명", "clear": "투명", "흰색": "하양", "흰": "하양",
             # 작은 모델이 가끔 중국어로 답해서 같이 처리
             "透明": "투명", "橘": "주황", "橙": "주황", "红": "빨강", "紅": "빨강", "棕": "갈색", "褐": "갈색",
             "白": "하양", "黄": "노랑", "黃": "노랑", "绿": "초록", "綠": "초록", "蓝": "파랑", "藍": "파랑",
             "紫": "보라", "粉": "분홍", "黑": "검정", "灰": "회색"}
SHAPE_MAP = {"oval": "타원형", "elliptical": "타원형", "椭圆": "타원형", "round": "원형", "circle": "원형",
             "circular": "원형", "圆": "원형", "oblong": "장방형", "capsule": "장방형",
             "rectangular": "장방형", "长方": "장방형", "triangle": "삼각형", "square": "사각형",
             "diamond": "마름모형", "pentagon": "오각형", "hexagon": "육각형", "octagon": "팔각형",
             "semicircle": "반원형", "橢圓": "타원형", "圓": "원형", "長方": "장방형"}

# 경계가 애매한 색·모양은 부분 점수 (예: 붉은 갈색 알약을 "빨강"이라고 답하는 경우)
COLOR_NEIGHBORS = {"갈색": {"빨강", "주황"}, "빨강": {"갈색", "주황", "분홍"}, "주황": {"빨강", "노랑", "갈색"},
                   "노랑": {"주황", "연두"}, "연두": {"노랑", "초록"}, "초록": {"연두", "청록"},
                   "청록": {"초록", "파랑"}, "파랑": {"청록", "남색"}, "남색": {"파랑", "보라"},
                   "보라": {"남색", "자주"}, "자주": {"보라", "분홍"}, "분홍": {"자주", "빨강"},
                   "하양": {"회색"}, "회색": {"하양", "검정"}, "검정": {"회색"}}
SHAPE_NEIGHBORS = {"원형": {"타원형"}, "타원형": {"원형", "장방형"}, "장방형": {"타원형"}}
NEIGHBOR_SCORE = 0.5

# ---------------------------------------------------------------- 전역 상태
S = {}          # load() 가 채움


def _find_col(df, cands):
    return next((c for c in cands if c in df.columns), None)


def _split_vals(v):
    return [x.strip() for x in re.split(r"[,|/]", str(v)) if x.strip()]


def norm_imp(s):
    s = str(s or "").upper()
    for w in ["십자분할선", "분할선", "마크", "로고"]:
        s = s.replace(w, " ")
    return re.sub(r"[^0-9A-Z가-힣]", "", s)


def _norm_attr(v, mapping, allowed):
    v = str(v or "").strip().lower()
    if not v:
        return ""
    for a in sorted(allowed, key=len, reverse=True):       # "타원형"을 "원형"보다 먼저
        if a in v:
            return a
    for k, t in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if k in v:
            return t
    return ""


def _postprocess(out):
    out = dict(out or {})
    out["color"] = _norm_attr(out.get("color"), COLOR_MAP, S["colors"])
    out["shape"] = _norm_attr(out.get("shape"), SHAPE_MAP, S["shapes"])
    out["imprints"] = [x for x in (out.get("imprints") or []) if norm_imp(x)]
    out["product_name"] = str(out.get("product_name") or "")
    return out


def load_photo(x):
    """경로 / PIL 이미지 / 업로드된 파일 객체 → 회전 보정된 RGB 이미지"""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    img = x if isinstance(x, Image.Image) else Image.open(x)
    return ImageOps.exif_transpose(img).convert("RGB")


def _shrink(img, max_side=1536):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img


# ---------------------------------------------------------------- 로드
def load(base_dir, config_path=None, device=None):
    """마스터, SigLIP, FAISS 인덱스, VLM을 불러와요. 앱 시작 시 한 번만 호출."""
    import faiss
    import torch
    from transformers import AutoModel, AutoProcessor

    index_dir = os.path.join(base_dir, "index")
    cfg = dict(DEFAULT_CONFIG)
    config_path = config_path or os.path.join(index_dir, "identify_config.json")
    if os.path.exists(config_path):
        cfg.update(json.load(open(config_path, encoding="utf-8")))
    S["cfg"] = cfg

    # 마스터
    master = pd.read_parquet(os.path.join(base_dir, "master.parquet"))
    key = _find_col(master, KEY_COL_CANDIDATES)
    name = _find_col(master, NAME_COL_CANDIDATES)
    master[key] = master[key].astype(str)
    m = master.drop_duplicates(key).set_index(key)
    cols = {k: _find_col(m, v) for k, v in COL_CANDIDATES.items()}
    S.update(m=m, cols=cols, items=m.index.tolist(),
             name_of=m[name].astype(str).to_dict())

    def val(item, k):
        c = cols.get(k)
        if not c or item not in m.index:
            return ""
        v = m.at[item, c]
        return "" if pd.isna(v) else str(v).strip()
    S["val"] = val

    S["shapes"] = sorted(m[cols["shape"]].dropna().astype(str).str.strip().unique()) if cols["shape"] else []
    S["colors"] = sorted({c for k in ["color1", "color2"] if cols[k]
                          for v in m[cols[k]].dropna() for c in _split_vals(v)})
    S["item_imps"] = {it: sorted({x for x in [norm_imp(val(it, "front")), norm_imp(val(it, "back"))] if x})
                      for it in S["items"]}
    S["item_names"] = {it: re.sub(r"\s", "", S["name_of"].get(it, "")) for it in S["items"]}

    # SigLIP + FAISS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    info = json.load(open(os.path.join(index_dir, "info.json")))
    if info["model"] != cfg["siglip_model"]:
        print(f"[identify] 설정의 SigLIP({cfg['siglip_model']}) 대신 인덱스 모델({info['model']})을 사용해요.")
        cfg["siglip_model"] = info["model"]
    S["torch"], S["device"], S["dtype"] = torch, device, dtype
    import gc
    S["siglip"] = AutoModel.from_pretrained(cfg["siglip_model"], torch_dtype=dtype, low_cpu_mem_usage=True,
                                            device_map=device if device == "cuda" else None).eval()
    gc.collect()
    S["siglip_proc"] = AutoProcessor.from_pretrained(cfg["siglip_model"])
    S["index"] = faiss.read_index(os.path.join(index_dir, "pill.faiss"))
    meta = pd.read_json(os.path.join(index_dir, "pill_meta.json"), dtype={"item_seq": str})
    S["vec_items"] = meta["item_seq"].to_numpy()

    # VLM
    _load_vlm(cfg["vlm_model"])
    print(f"[identify] 준비 완료: 약 {len(S['items'])}개, 벡터 {S['index'].ntotal}개, "
          f"VLM {cfg['vlm_model']}, 가중치 {cfg['weights']}")


def _build_prompt():
    return f'''You are looking at a user's photo of a medicine (a loose pill, a blister pack, or a box).
Return ONLY a JSON object with these keys:
- "product_name": the product name printed on a box or blister foil if clearly readable, otherwise "".
- "imprints": list of the letters/numbers engraved or printed on the pill surface, exactly as seen (e.g. ["TYLENOL", "500"]). [] if none are visible. Never guess.
- "color": the main pill color, one of {S["colors"]}. "" if no pill is visible.
- "shape": the pill shape, one of {S["shapes"]}. "" if no pill is visible.
- "box_2d": bounding box of the most visible pill as [ymin, xmin, ymax, xmax] normalized to 0-1000, or null.
'''


def _load_vlm(model_name):
    S["prompt"] = _build_prompt()
    if model_name.startswith("gemini"):
        from google import genai
        from google.genai import types
        S["genai_client"] = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        S["genai_types"] = types
        S["vlm_kind"] = "gemini"
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        torch = S["torch"]
        S["qwen"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if S["device"] == "cuda" else torch.float32,
            device_map=S["device"], low_cpu_mem_usage=True).eval()
        S["qwen_proc"] = AutoProcessor.from_pretrained(model_name, min_pixels=256 * 28 * 28,
                                                       max_pixels=1280 * 28 * 28)
        S["vlm_kind"] = "qwen"


def _parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        out = json.loads(m.group(0))
        if isinstance(out, list):
            out = out[0] if out else {}
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _vlm_gemini(img, retries=3):
    types = S["genai_types"]
    models = [S["cfg"]["vlm_model"]] + list(S["cfg"].get("vlm_fallbacks") or [])
    for model_name in models:
        for a in range(retries):
            try:
                resp = S["genai_client"].models.generate_content(
                    model=model_name, contents=[_shrink(img), S["prompt"]],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                return _parse_json(resp.text)
            except Exception as e:
                msg = str(e)
                print(f"[identify] {model_name} 재시도 {a + 1}: {msg[:80]}")
                if any(c in msg for c in ("404", "403", "429")):   # 없음·권한·하루 한도 초과는 기다려도 소용없음
                    break
                time.sleep(5 * (a + 1))
    return {}


def _vlm_qwen(img):
    torch = S["torch"]
    proc, model = S["qwen_proc"], S["qwen"]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": S["prompt"]}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[_shrink(img, 1280)], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    out = _parse_json(proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
    out.pop("box_2d", None)          # Qwen은 좌표 방식이 달라서 사용 안 함
    return out


def vlm_extract(img):
    """사진 → {"product_name", "imprints", "color", "shape", "box_2d"} (실패 시 빈 값)"""
    try:
        raw = _vlm_gemini(img) if S["vlm_kind"] == "gemini" else _vlm_qwen(img)
    except Exception as e:
        print(f"[identify] VLM 실패, SigLIP만 사용: {e}")
        raw = {}
    return _postprocess(raw) if raw else {}


# ---------------------------------------------------------------- SigLIP 검색
def _embed(images):
    torch = S["torch"]
    with torch.no_grad():
        inputs = S["siglip_proc"](images=[im.convert("RGB") for im in images], return_tensors="pt").to(S["device"])
        inputs["pixel_values"] = inputs["pixel_values"].to(S["dtype"])
        f = S["siglip"].get_image_features(**inputs)
        f = (f if torch.is_tensor(f) else f.pooler_output).float()
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype("float32")


def _search(img, k=POOL, n_vec=150):
    scores, idxs = S["index"].search(_embed([img]), n_vec)
    best = {}
    for s, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        it = S["vec_items"][i]
        best[it] = max(best.get(it, -1.0), float(s))
    return dict(sorted(best.items(), key=lambda x: -x[1])[:k])


def _crop_box(img, box, pad=0.15):
    try:
        y0, x0, y1, x1 = [float(v) / 1000 for v in box]
    except Exception:
        return None
    W, H = img.size
    bw, bh = (x1 - x0) * W, (y1 - y0) * H
    if bw < 20 or bh < 20:
        return None
    return img.crop((int(max(0, x0 * W - bw * pad)), int(max(0, y0 * H - bh * pad)),
                     int(min(W, x1 * W + bw * pad)), int(min(H, y1 * H + bh * pad))))


def siglip_scores(img, vlm=None):
    views = [img]
    if vlm and vlm.get("box_2d"):
        c = _crop_box(img, vlm["box_2d"])
        if c is not None:
            views.append(c)
    sig = {}
    for v in views:
        for it, s in _search(v).items():
            sig[it] = max(sig.get(it, -1.0), s)
    return sig


# ---------------------------------------------------------------- 점수 결합
def _imprint_score(vlm_imps, item):
    from rapidfuzz import fuzz
    targets = S["item_imps"].get(item, [])
    q = [x for x in (norm_imp(v) for v in (vlm_imps or [])) if x]
    if not q or not targets:
        return 0.0
    qs, ts = q + ["".join(q)], targets + ["".join(targets)]
    return max(fuzz.ratio(a, b) for a in qs for b in ts) / 100


def _attr_score(vlm, item):
    val = S["val"]
    s, n = 0.0, 0
    if vlm.get("color"):
        n += 1
        gold = set(_split_vals(val(item, "color1") + "," + val(item, "color2")))
        c = vlm["color"]
        s += 1.0 if c in gold else (NEIGHBOR_SCORE if gold & COLOR_NEIGHBORS.get(c, set()) else 0.0)
    if vlm.get("shape") and S["cols"]["shape"]:
        n += 1
        g, sh = val(item, "shape"), vlm["shape"]
        s += 1.0 if sh == g else (NEIGHBOR_SCORE if g in SHAPE_NEIGHBORS.get(sh, set()) else 0.0)
    return s / n if n else 0.0


def _name_matches(product_name, limit=10):
    from rapidfuzz import fuzz, process
    q = re.sub(r"\s", "", str(product_name or ""))
    if len(q) < 2:
        return {}
    res = process.extract(q, S["item_names"], scorer=fuzz.partial_ratio, limit=limit)
    return {key: score / 100 for (_, score, key) in res}


def rank_candidates(sig, vlm, k=3, weights=None):
    """SigLIP 점수 + VLM 정보 → [(item_seq, score, parts)] 상위 k개"""
    vlm = vlm or {}
    if not vlm:                                         # VLM 실패 시 SigLIP만
        return [(it, s, {"siglip": s}) for it, s in sorted(sig.items(), key=lambda x: -x[1])[:k]]

    w = dict(weights or S["cfg"]["weights"])
    th = S["cfg"]["name_threshold"]
    imps = vlm.get("imprints") or []
    imp_all = {it: _imprint_score(imps, it) for it in S["items"]} if imps else {}
    names = _name_matches(vlm.get("product_name"))
    if not imps:                                        # 각인 못 읽으면 SigLIP에 몰아주기
        w["siglip"] += w["imprint"]
        w["imprint"] = 0.0

    imp_top = sorted(imp_all.items(), key=lambda x: -x[1])[:POOL]
    cand = set(sig) | {it for it, s in imp_top if s >= 0.6} | {it for it, s in names.items() if s >= th}
    lo, hi = (min(sig.values()), max(sig.values())) if sig else (0.0, 1.0)

    rows = []
    for it in cand:
        parts = {"siglip": (sig.get(it, lo) - lo) / (hi - lo + 1e-6),
                 "imprint": imp_all.get(it, 0.0),
                 "attr": _attr_score(vlm, it)}
        score = sum(w[key] * parts[key] for key in w)
        if names.get(it, 0) >= th:
            parts["name"] = names[it]
            score += NAME_BONUS * names[it]
        rows.append((it, score, parts))
    rows.sort(key=lambda x: -x[1])
    return rows[:k]


# ---------------------------------------------------------------- 공개 함수
def identify(image, k=3, return_debug=False):
    """알약 사진 → [{"item_seq", "name", "image_url", "score"}] (score 높은 순 k개)"""
    if not S:
        raise RuntimeError("identify.load(base_dir)를 먼저 호출하세요.")
    img = load_photo(image)
    vlm = vlm_extract(img)
    sig = siglip_scores(img, vlm)
    rows = rank_candidates(sig, vlm, k=k)
    out = [{"item_seq": it, "name": S["name_of"].get(it, ""), "image_url": S["val"](it, "image"),
            "score": round(float(s), 4)} for it, s, _ in rows]
    return (out, {"vlm": vlm, "parts": rows}) if return_debug else out
