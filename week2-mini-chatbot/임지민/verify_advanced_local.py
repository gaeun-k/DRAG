"""Optional real local-model smoke check; no paid services."""
import json
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from advanced_rag import StudyRag, Options, cluster_labels

if __name__ == "__main__":
    docs = [Document(page_content=text, metadata={"source": f"검증문서{i}.txt"})
            for i, text in enumerate([
                "스터디는 매주 수요일 저녁 7시에 열린다. 장소는 온라인 회의실이다.",
                "과제 제출물은 코드, 원본 문서, 실행 설명서다. 제출 기한은 화요일이다.",
                "실습은 로컬 모델을 사용한다. API 키나 결제는 필요하지 않다."])]
    bot = StudyRag(docs, Options(raptor=True, search="multiquery"))
    result = {"summary_nodes": len(bot.indexed) - len(bot.chunks),
              "multiquery_answer": bot.ask("모임은 언제 어디에서 열리나요?"),
              "density_rounds": bot.density_summary()}
    groups = cluster_labels(np.random.default_rng(42).normal(size=(12, 24)))
    assert set().union(*map(set, groups)) == set(range(12))
    assert result["summary_nodes"] >= 1 and len(result["density_rounds"]) == 5
    result["umap_group_sizes"] = list(map(len, groups))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/advanced-local-verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Local RAPTOR, MultiQuery, CoD 5 rounds, UMAP/GMM: completed", flush=True)
