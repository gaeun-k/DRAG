"""Explicit real local-model smoke test; no paid APIs, results go to outputs/."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
from advanced_rag import Options, StudyRag
from ingest import load_file, load_url
from local_ai import describe_image

root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)
docs = load_file("study_wiki.md", (root / "data/study_wiki.md").read_bytes())
bot = StudyRag(docs, Options(mode="local"))
results = {}
for question in ["2주차 과제 제출물은 무엇인가요?", "그중 문서에 없는 질문도 포함해야 하나요?", "과제 마감 날짜는 언제인가요?"]:
    results[question] = bot.ask(question)
    (out / "local-verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Completed text question", flush=True)
image = Image.new("RGB", (600, 250), "white")
draw = ImageDraw.Draw(image)
draw.text((30, 40), "Product   Quantity\nApple      3\nPear       7", fill="black", font_size=36)
path = out / "vision-test.png"
image.save(path)
visual_docs = load_file("vision-test.png", path.read_bytes(), vision=True, describe=describe_image)
visual_bot = StudyRag(visual_docs, Options(mode="local"))
results["vision"] = visual_bot.ask("표에서 Pear의 수량은 얼마인가요?")
(out / "local-verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("Completed image question", flush=True)
results["web"] = [{"source": d.metadata["source"], "characters": len(d.page_content)} for d in load_url("https://example.com")]
(out / "local-verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved outputs/local-verification.json", flush=True)
