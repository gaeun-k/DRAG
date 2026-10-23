"""약 정보 질의응답 (A: RAG)

사용법
    import rag
    rag.load(BASE)                               # 앱 시작 시 한 번 (BASE 안에 master.parquet, efficacy_emb.npz)
    rag.search("콧물이 나요")                     # 증상 질문
    rag.search("공복에 먹어도 돼?", item_seq="...")  # 식별된 약에 대한 질문
    # -> {"answer": str, "sources": [{"item_seq", "name", "section"}]}
GOOGLE_API_KEY 환경변수(또는 .env)가 필요해요.
"""

import os
import re
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, ".env"))
    load_dotenv(os.path.join(HERE, "..", ".env"))
except ImportError:
    pass

MASTER_PATH = os.path.join(HERE, "master.parquet")
CHUNKS_PATH = os.path.join(HERE, "chunks.parquet")
EMB_PATH = os.path.join(HERE, "efficacy_emb.npz")

EMBED_MODEL = os.getenv("MEDBOT_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("MEDBOT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODELS = [os.getenv("MEDBOT_LLM_MODEL", "gemini-flash-latest"), "gemini-flash-lite-latest"]

# master 컬럼 -> 사용자에게 보여줄 섹션명
SECTIONS = {
    "efficacy": "효능",
    "use_method": "복용법",
    "warning": "경고",
    "precaution": "주의사항",
    "interaction": "상호작용",
    "side_effect": "부작용",
    "storage": "보관법",
}

# 증상 질문에서 약 추천 대신 병원 안내를 우선할 표현
DANGER_KEYWORDS = [
    "고열", "39도", "40도", "흉통", "가슴 통증", "가슴이 아", "가슴이 답답", "호흡곤란", "숨이 차", "숨쉬기 힘",
    "숨을 쉴 수", "의식", "실신", "기절", "경련", "발작", "혈변", "토혈", "피를 토", "피가 섞", "검은 변",
    "심한 복통", "마비", "말이 어눌", "얼굴이 한쪽", "자살", "죽고 싶", "아나필락시스", "입술이 붓", "얼굴이 붓",
    "전신 두드러기",
]

# 약을 선택한 상태에서, 질문이 '그 약'에 대한 것인지 알아보는 표현
ITEM_CUES = (
    "이 약", "이약", "이거", "이건", "이걸", "해당 약", "복용", "먹", "드시", "드셔", "마셔", "섭취", "용법", "용량",
    "몇", "언제", "식전", "식후", "공복", "하루", "부작용", "주의", "금기", "임신", "임부", "수유", "어린이", "아이",
    "소아", "노인", "술", "음주", "알코올", "같이", "함께", "병용", "보관", "유통기한", "효능", "효과", "성분",
)

SCOPE_NOTICE = (
    "※ 이 서비스는 낱알 식별이 가능한 일반의약품(정제·캡슐 등)만 다뤄요. "
    "같은 이름의 액상·시럽 등 다른 제형 제품은 포함되어 있지 않아요."
)
DANGER_ANSWER = (
    "말씀하신 증상은 약으로 해결하기보다 의료진의 진료가 필요할 수 있어요. "
    "증상이 심하거나 갑자기 나타났다면 가까운 병원 응급실을 방문하거나 119에 연락하세요."
)


# ---------------------------------------------------------------- 청킹
def clean_text(s):
    if s is None or pd.isna(s):
        return ""
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def build_chunks(master_path=MASTER_PATH, out_path=CHUNKS_PATH):
    """item_seq 단위로 중복을 없애고 (품목, 섹션) 하나를 청크 하나로 만든다."""
    master = pd.read_parquet(master_path).drop_duplicates("item_seq")
    rows = []
    for r in master.itertuples(index=False):
        for col, label in SECTIONS.items():
            text = clean_text(getattr(r, col))
            if not text:
                continue
            rows.append({
                "item_seq": r.item_seq,
                "name": r.item_name,
                "class_name": clean_text(r.class_name),
                "etc_otc": clean_text(r.etc_otc_name),
                "form": clean_text(r.form_code_name),
                "section": label,
                "text": text,
                # 검색용: 품목명과 섹션명을 앞에 붙여 이름·섹션 질의도 걸리게 한다
                "search_text": f"{r.item_name} {label} {text}",
            })
    chunks = pd.DataFrame(rows)
    chunks.insert(0, "chunk_id", range(len(chunks)))
    chunks.to_parquet(out_path, index=False)
    return chunks


# ---------------------------------------------------------------- 키워드 검색 (BM25)
_KIWI = None
_KEEP = ("NNG", "NNP", "VV", "VA", "XR", "SL", "SN")


def tokenize(text):
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    return [t.form for t in _KIWI.tokenize(text) if t.tag.startswith(_KEEP)]


class Bm25Index:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks.reset_index(drop=True)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.chunks["search_text"]])

    def scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)))


# ---------------------------------------------------------------- 임베딩 검색
class DenseIndex:
    def __init__(self, chunks, model_name=EMBED_MODEL, cache_path=EMB_PATH):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        ids = self.chunks["chunk_id"].to_numpy()
        emb = None
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            if np.array_equal(z["ids"], ids) and str(z["model"]) == model_name:
                emb = z["emb"]
        if emb is None:
            emb = self.model.encode(self.chunks["search_text"].tolist(), batch_size=32,
                                    normalize_embeddings=True, show_progress_bar=True)
            np.savez(cache_path, emb=emb, ids=ids, model=model_name)
        self.emb = emb.astype("float32")

    def scores(self, query):
        q = self.model.encode([query], normalize_embeddings=True)[0].astype("float32")
        return self.emb @ q


def _rrf(rank_lists, weights=None, k=60):
    weights = weights or [1.0] * len(rank_lists)
    out = {}
    for ranks, w in zip(rank_lists, weights):
        for r, i in enumerate(ranks):
            out[i] = out.get(i, 0.0) + w / (k + r + 1)
    return out


# ---------------------------------------------------------------- 재순위 (reranker)
class Reranker:
    def __init__(self, model_name=RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)

    def scores(self, query, texts):
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=16))


# ---------------------------------------------------------------- 검색기
class Retriever:
    def __init__(self, chunks_path=CHUNKS_PATH, emb_path=EMB_PATH, use_dense=True, use_rerank=True):
        self.chunks = pd.read_parquet(chunks_path)
        self.by_item = {k: g for k, g in self.chunks.groupby("item_seq")}
        eff = self.chunks[self.chunks["section"] == "효능"].reset_index(drop=True)
        self.eff = eff
        self.bm25 = Bm25Index(eff)
        self.dense = DenseIndex(eff, cache_path=emb_path) if use_dense else None
        self.reranker = Reranker() if use_rerank else None
        self._build_name_index()

    # 품목명에서 브랜드 줄기(예: 타이레놀정500밀리그람(아세트아미노펜) -> 타이레놀)를 뽑는다
    @staticmethod
    def _stem(name):
        s = re.sub(r"\(.*?\)", "", str(name))
        s = re.split(r"[0-9]", s)[0]
        s = re.sub(r"(연질캡슐|경질캡슐|캡슐|서방정|장용정|츄어블정|정|시럽|현탁액|액|산|과립|환)$", "", s.strip())
        return re.sub(r"\s", "", s)

    def _build_name_index(self):
        names = self.chunks.drop_duplicates("item_seq")[["item_seq", "name"]]
        self.stems = {}
        for it, nm in zip(names["item_seq"], names["name"]):
            st = self._stem(nm)
            if len(st) >= 2:
                self.stems.setdefault(st, []).append(it)

    def resolve_items(self, question, max_items=6):
        """질문 속 약 이름을 item_seq 목록으로 바꾼다. 못 찾거나 너무 모호하면 []."""
        q = re.sub(r"\s", "", question)
        hit = [st for st in self.stems if st in q]
        hit = [st for st in hit if not any(st != o and st in o for o in hit)]   # 더 긴 이름 우선
        if not hit:   # 정확한 이름이 없을 때만, 제품명 앞부분 매칭 (예: 판피린 -> 판피린티정)
            words = {w for w in tokenize(question) if len(w) >= 3}
            hit = [st for st in self.stems if any(st.startswith(w) for w in words)]
        items = [it for st in hit for it in self.stems[st]]
        return items if 0 < len(items) <= max_items else []

    def item_chunks(self, item_seq):
        g = self.by_item.get(str(item_seq))
        return g if g is not None else self.chunks.iloc[0:0]

    def symptom_items(self, query, k=3, pool=30, weights=(0.5, 1.0), rerank=True, rerank_pool=40):
        """증상 질의 -> 효능 청크 하이브리드(BM25 + 임베딩, 가중 RRF) -> (reranker 재정렬) -> 상위 k개 품목."""
        lists, ws = [], []
        for idx, w in zip((self.bm25, self.dense), weights):
            if idx is None:
                continue
            s = idx.scores(query)
            order = np.argsort(-s)[:pool]
            lists.append([int(i) for i in order if s[i] > 0])
            ws.append(w)
        fused = _rrf(lists, ws)
        top = sorted(fused, key=lambda i: -fused[i])
        if rerank and self.reranker is not None and top:
            cands, seen_items = [], set()
            for i in top:                       # 후보: 같은 약은 한 번만, 상위 rerank_pool 개
                it = self.eff.at[i, "item_seq"]
                if it not in seen_items:
                    seen_items.add(it)
                    cands.append(i)
                if len(cands) == rerank_pool:
                    break
            rs = self.reranker.scores(query, [self.eff.at[i, "text"] for i in cands])
            top = [cands[j] for j in np.argsort(-rs)]
        out, seen, sigs = [], set(), []
        for i in top:
            it = self.eff.at[i, "item_seq"]
            if it in seen:
                continue
            seen.add(it)
            sig = set(tokenize(self.eff.at[i, "text"]))
            if any(len(sig & s) / max(1, len(sig | s)) >= 0.85 for s in sigs):   # 효능 문구가 거의 같은 약은 건너뜀
                continue
            sigs.append(sig)
            out.append(it)
            if len(out) == k:
                break
        return out


# ---------------------------------------------------------------- LLM
_CLIENT = None


def _generate(prompt):
    global _CLIENT
    from google import genai
    from google.genai import types
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    for model in LLM_MODELS:
        for attempt in range(3):
            try:
                r = _CLIENT.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2))
                if r.text:
                    return r.text.strip()
            except Exception as e:
                msg = str(e)
                if any(c in msg for c in ("404", "403")):
                    break
                time.sleep(3 * (attempt + 1))
    return None


def _build_prompt(question, context, mode):
    if mode == "item":
        task = ("사용자가 특정 약에 대해 물었습니다. 질문에 맞는 내용을 우선 답하고, 질문이 일반적이면 "
                "이름, 분류, 복용법, 주의사항 순서로 정리하세요. 자료에 제품이 여러 개면 제품별로 나눠서 쓰세요.")
        extra = ""
    else:
        task = ("사용자가 증상을 말했습니다. 아래 자료에서 증상에 맞는 일반의약품을 2~3개 예시로 소개하고 "
                "각각의 공식 효능을 요약하세요. 가장 적합하다고 단정하지 말고 예시로만 제시하세요.")
        extra = ("\n- 약마다 [주의사항]·[경고]에서 복용 전 꼭 확인할 점을 1~2개 \"복용 전 확인\"으로 덧붙이세요."
                 "\n- 마지막에 \"증상의 원인에 따라 맞는 약이 다를 수 있어요\"를 한 줄 쓰세요.")
    return f"""당신은 식품의약품안전처 의약품개요정보를 안내하는 도우미입니다.
{task}

규칙:
- 아래 [자료]에 있는 내용만 사용하고, 자료에 없으면 "제공된 정보에는 없어요"라고 답하세요.
- 진단하거나 처방하는 말투를 쓰지 말고 "식약처 정보에 따르면 ..."처럼 안내하세요.
- 약 이름을 언급할 때 어떤 항목(효능, 복용법, 주의사항 등)에서 가져왔는지 자연스럽게 밝히세요.
- 제품(제형)이 다르면 내용을 한 문장으로 묶지 말고 제품별로 따로 쓰세요. 내용이 같아도 제품마다 각각 적으세요.
- 한국어로 간결하게, 문단 대신 짧은 목록을 사용하세요. 면책 문구는 따로 붙이니 쓰지 마세요.{extra}

[자료]
{context}

[질문]
{question}
"""


def _context_for(chunks_df, sections=None):
    parts = []
    for it, g in chunks_df.groupby("item_seq", sort=False):
        form = g.iloc[0]["form"] if "form" in g.columns else ""
        head = f"## {g.iloc[0]['name']} (제형: {form}, 분류: {g.iloc[0]['class_name']}, {g.iloc[0]['etc_otc']})"
        body = [f"[{r.section}] {r.text}" for r in g.itertuples() if not sections or r.section in sections]
        parts.append(head + "\n" + "\n".join(body))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 공개 함수
_R = None


def load(base_dir=HERE, use_dense=True, use_rerank=None):
    """base_dir 안의 master.parquet, efficacy_emb.npz 를 사용 (chunks.parquet 은 없으면 자동 생성)
    use_rerank: 기본은 켜짐. GPU 가 없어 느리면 환경변수 MEDBOT_RERANK=0 으로 끈다."""
    global _R
    if use_rerank is None:
        use_rerank = os.getenv("MEDBOT_RERANK", "1") != "0"
    chunks_path = os.path.join(base_dir, "chunks.parquet")
    if not os.path.exists(chunks_path) or "form" not in pd.read_parquet(chunks_path).columns:
        build_chunks(os.path.join(base_dir, "master.parquet"), chunks_path)
    _R = Retriever(chunks_path, os.path.join(base_dir, "efficacy_emb.npz"), use_dense, use_rerank)
    return _R


def _asks_about_item(question):
    return any(c in question for c in ITEM_CUES)


def _bullets(text):
    return "\n".join(f"- {ln.strip()}" for ln in str(text).split("\n") if ln.strip())


def _template_answer(r, item_seq):
    """LLM 없이 이름·종류·복용법·주의사항을 고정 형식으로 보여준다 (사진으로 약을 고른 직후용)."""
    g = r.item_chunks(item_seq)
    if g.empty:
        return {"answer": "제공 범위(일반의약품 2,744종) 안에서 해당 약의 정보를 찾지 못했어요.", "sources": []}
    head = g.iloc[0]
    text = {row.section: row.text for row in g.itertuples()}
    kind = " · ".join(x for x in (head["class_name"], head["etc_otc"], head.get("form", "")) if x)
    lines = [f"**{head['name']}**", f"- **종류**: {kind}"]
    used = []
    for section in ("복용법", "경고", "주의사항"):
        body = text.get(section)
        if body:
            used.append(section)
            lines += ["", f"**{section}**", _bullets(body)]
        elif section != "경고":
            lines += ["", f"**{section}**", "- 제공된 정보에는 없어요"]
    sources = [{"item_seq": head["item_seq"], "name": head["name"], "section": s} for s in used]
    return {"answer": "\n".join(lines), "sources": sources}


def search(question=None, item_seq=None):
    """question(str|None), item_seq(str, 선택) -> {"answer": str, "sources": [{item_seq, name, section}]}

    question 이 없고 item_seq 만 있으면 LLM 없이 고정 템플릿으로 답한다.
    item_seq 가 있어도 질문이 그 약과 무관한 증상 질문이면 전체 검색으로 답한다."""
    if _R is None:
        load()
    r = _R

    question = (question or "").strip()
    if not question:
        if not item_seq:
            return {"answer": "증상이나 궁금한 점을 입력해 주세요.", "sources": []}
        return _template_answer(r, item_seq)

    if any(k in question for k in DANGER_KEYWORDS):
        return {"answer": DANGER_ANSWER, "sources": []}

    by_name, switched_from = False, None
    named = r.resolve_items(question)
    if item_seq and (not named or set(named) == {str(item_seq)}):
        if _asks_about_item(question):
            items, mode = [str(item_seq)], "item"
        else:
            sel = r.item_chunks(item_seq)
            switched_from = sel.iloc[0]["name"] if not sel.empty else "선택한 약"
            items, mode = r.symptom_items(question), "symptom"
    elif item_seq:       # 선택한 약 + 질문에서 언급한 다른 약 (예: 병용 질문)
        items, mode, by_name = [str(item_seq)] + [it for it in named if it != str(item_seq)], "item", True
    elif named:
        items, mode, by_name = named, "item", True
    else:
        items, mode = r.symptom_items(question), "symptom"

    if not items:
        return {"answer": "관련된 일반의약품 정보를 찾지 못했어요. 증상을 조금 더 구체적으로 알려주세요.", "sources": []}

    if mode == "item":
        sections = None
    else:
        sections = ["효능", "복용법", "주의사항", "경고"]
    ctx_df = pd.concat([r.item_chunks(it) for it in items])
    if sections:
        ctx_df = ctx_df[ctx_df["section"].isin(sections)]
    if ctx_df.empty:
        return {"answer": "제공 범위(일반의약품 2,744종) 안에서 해당 약의 정보를 찾지 못했어요.", "sources": []}

    answer = _generate(_build_prompt(question, _context_for(ctx_df), mode))
    if answer is None:   # LLM 실패 시 원문을 그대로 보여준다
        answer = "답변 생성에 실패해 식약처 원문을 그대로 보여드려요.\n\n" + _context_for(ctx_df)
    cited = _cited_items(ctx_df, answer)
    sources = [{"item_seq": x.item_seq, "name": x.name, "section": x.section}
               for x in ctx_df[ctx_df["item_seq"].isin(cited)].itertuples()]
    notice = f"\n\n{SCOPE_NOTICE}" if by_name else ""
    if switched_from:
        notice += f"\n\n※ 선택하신 약({switched_from})이 아니라, 말씀하신 증상을 기준으로 찾은 약이에요."
    return {"answer": f"{answer}{notice}", "sources": sources}


def _cited_items(ctx_df, answer):
    """답변에 실제로 이름이 나온 약만 sources 로 남긴다 (하나도 없으면 전체)."""
    keep = []
    for it, g in ctx_df.groupby("item_seq", sort=False):
        name = g.iloc[0]["name"]
        stem = Retriever._stem(name)
        if re.sub(r"\(.*?\)", "", name) in answer or (len(stem) >= 2 and stem in answer):
            keep.append(it)
    return keep or list(ctx_df["item_seq"].unique())
