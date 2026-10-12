"""BM25 후보 검색 + 선택 Ollama 임베딩/FAISS 검색. 서로 다른 자료 종류를 섞지 않습니다."""
from collections import Counter
import math
import re


def tokens(text):
    text = text.lower()
    result = re.findall(r"[a-z0-9][a-z0-9+#.]*|[가-힣]+", text)
    # 형태소 라이브러리 없이 한국어 조사·어미 차이를 일부 흡수합니다.
    for word in re.findall(r"[가-힣]{2,}", text):
        result += [word[i:i + 2] for i in range(len(word) - 1)]
    return result


def lexical_scores(query, texts):
    counts = [Counter(tokens(text)) for text in texts]
    if not counts:
        return []
    mean = sum(sum(c.values()) for c in counts) / len(counts) or 1
    terms = set(tokens(query))
    frequencies = {term: sum(term in c for c in counts) for term in terms}
    scores = []
    for count in counts:
        value = 0.0
        length = sum(count.values())
        for term in terms:
            freq = count[term]
            if freq:
                inverse = math.log(1 + (len(counts) - frequencies[term] + .5) /
                                   (frequencies[term] + .5))
                value += inverse * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / mean))
        scores.append(value)
    return scores


def normalized(values):
    import numpy as np
    vectors = np.asarray(values, dtype="float32")
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("임베딩이 올바르지 않습니다.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("0 벡터는 검색할 수 없습니다.")
    return np.ascontiguousarray(vectors / norms, dtype="float32")


def search(query, records, k=5, ai=None, store=None):
    if not query.strip() or not records:
        return []
    if k < 1:
        raise ValueError("검색 개수는 1 이상이어야 합니다.")
    scores = lexical_scores(query, [r["text"] for r in records])
    lexical = sorted((i for i, score in enumerate(scores) if score > 0),
                     key=lambda i: scores[i], reverse=True)
    if ai is None:
        return [{**records[i], "score": scores[i], "search_method": "BM25"} for i in lexical[:k]]
    from .config import SETTINGS
    # embeddinggemma의 검색용 접두사도 캐시 버전에 포함합니다.
    cache_key = SETTINGS.embed_model + "|retrieval-prompts-v1"
    cache = store.get_vectors(cache_key) if store else {}
    missing = [r for r in records if r["id"] not in cache]
    for start in range(0, len(missing), 8):
        batch = missing[start:start + 8]
        values = ai.embed(["title: " + r.get("name", "") + " | text: " + r["text"] for r in batch])
        pairs = [(r["id"], vector) for r, vector in zip(batch, values)]
        cache.update(pairs)
        if store:
            store.save_vectors(cache_key, pairs)
    vectors = normalized([cache[r["id"]] for r in records])
    query_vector = normalized(ai.embed(["task: search result | query: " + query]))
    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    similarities, ids = index.search(query_vector, min(12, len(records)))
    dense = [int(i) for score, i in zip(similarities[0], ids[0]) if score > 0 and i >= 0]
    combined = {}
    for ranking in (lexical[:12], dense):
        for rank, i in enumerate(ranking, 1):
            combined[i] = combined.get(i, 0) + 1 / (60 + rank)
    ordered = sorted(combined, key=combined.get, reverse=True)[:k]
    return [{**records[i], "score": combined[i], "search_method": "BM25 + FAISS / RRF"} for i in ordered]


def source_label(hit):
    position = f"{hit['page']}쪽" if hit.get("page") else hit.get("section") or "본문"
    return f"{hit['name']} · {position}"
