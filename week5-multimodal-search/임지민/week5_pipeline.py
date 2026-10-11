"""Week 5: Flickr8k -> SigLIP 2 -> FAISS -> bilingual retrieval and Recall@K."""
# %% imports
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import faiss
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModel, AutoProcessor

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA = BASE / 'data'
RESULTS = BASE / 'results'
MODEL_ID = 'google/siglip2-base-patch16-224'
DATASET = 'intro/flickr8k'
REVISION = 'c7a02cb487da0eca729e133194b0f842cb578c08'
SEED = 42
N_IMAGES = 300
BATCH_SIZE = 8
DEVICE = 'cpu'  # Windows/Mac 모두 가능한 기본값. CUDA는 별도 설정 가능.
torch.set_num_threads(min(4, os.cpu_count() or 1))

# %% dataset
def download(url, destination):
    if destination.exists():
        return
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'DRAG-week5-study/1.0'}), timeout=45) as response:
                content = response.read()
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            temporary.write_bytes(content)
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare_dataset():
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    root = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/test/'
    download(root + 'metadata.csv', DATA / 'metadata.csv')
    with (DATA / 'metadata.csv').open(encoding='utf-8', newline='') as handle:
        all_rows = sorted(csv.DictReader(handle), key=lambda row: row['file_name'])
    selected = random.Random(SEED).sample(all_rows, N_IMAGES)
    records = []
    for row in selected:
        name = row['file_name']
        if Path(name).name != name:
            raise ValueError('Invalid dataset filename')
        records.append({'id': name, 'path': 'data/' + name,
                        'captions': [row[f'caption_{i}'] for i in range(5)],
                        'source': root + name})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda r: download(r['source'], BASE / r['path']), records))
    for record in records:
        with Image.open(BASE / record['path']) as image:
            image.verify()
        record['sha256'] = hashlib.sha256((BASE / record['path']).read_bytes()).hexdigest()
    assert len({r['id'] for r in records}) == N_IMAGES
    (RESULTS / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Dataset: {len(records)} images / {len(records)*5} captions / seed={SEED}', flush=True)
    return records

# %% model
def load_model():
    # 기존 프로젝트 캐시가 있으면 재사용. 다른 컴퓨터에서는 자동 다운로드.
    from huggingface_hub import snapshot_download
    cache = BASE.parents[1] / '.local-ai' / 'siglip'
    source = MODEL_ID
    if cache.exists():
        try:
            source = snapshot_download(MODEL_ID, cache_dir=str(cache), local_files_only=True)
        except OSError:
            pass
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=False)
    model = AutoModel.from_pretrained(source, trust_remote_code=False, use_safetensors=True).to(DEVICE).eval()
    return processor, model


def normalize(features):
    # FAISS는 contiguous float32 배열을 사용한다.
    array = np.ascontiguousarray(features.detach().cpu().float().numpy(), dtype=np.float32)
    faiss.normalize_L2(array)
    return array


def embed_images(records, processor, model):
    batches = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            images = []
            for record in records[start:start+BATCH_SIZE]:
                with Image.open(BASE / record['path']) as image:
                    images.append(ImageOps.exif_transpose(image).convert('RGB'))
            inputs = processor(images=images, return_tensors='pt').to(DEVICE)
            batches.append(normalize(model.get_image_features(**inputs)))
            print(f'Image embeddings {min(start+BATCH_SIZE,len(records))}/{len(records)}', flush=True)
    return np.concatenate(batches)


def embed_texts(texts, processor, model):
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            inputs = processor(text=texts[start:start+16], padding='max_length',
                               truncation=True, max_length=64, return_tensors='pt').to(DEVICE)
            batches.append(normalize(model.get_text_features(**inputs)))
    return np.concatenate(batches)

# %% index
def build_index(vectors):
    assert vectors.dtype == np.float32 and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    # 정규화된 벡터의 내적은 코사인 유사도. 높을수록 유사하다.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    # Windows 한글 경로 호환: FAISS 직접 경로 I/O 대신 serialize 사용.
    (RESULTS / 'images.faiss').write_bytes(faiss.serialize_index(index).tobytes())
    restored = faiss.deserialize_index(np.frombuffer((RESULTS / 'images.faiss').read_bytes(), dtype=np.uint8).copy())
    assert restored.ntotal == len(vectors)
    scores, ids = restored.search(vectors[:5], 1)
    assert np.array_equal(ids[:,0], np.arange(5))
    assert np.allclose(scores, 1, atol=1e-5)
    print(f'FAISS saved/reloaded: {restored.ntotal} vectors, dimension={restored.d}', flush=True)
    return restored


def search(query, index, records, processor, model, k=5):
    if not query.strip() or not 1 <= k <= len(records):
        raise ValueError('Query and k must be valid')
    vector = embed_texts([query], processor, model)
    scores, ids = index.search(vector, k)
    return [{'rank': rank+1, 'score': float(score), **records[int(i)]}
            for rank, (score, i) in enumerate(zip(scores[0], ids[0]))]

# %% result rendering
def font(size):
    candidates = [os.environ.get('KOREAN_FONT', ''), 'C:/Windows/Fonts/malgun.ttf',
                  '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                  '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.truetype('DejaVuSans.ttf', size)


def render_results(query, hits, filename, language):
    # 실제 FAISS 반환 결과를 캡처용 이미지로 저장한다. 순위 수동 교체 없음.
    canvas = Image.new('RGB', (1800, 790), '#f4f6fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((45,25), f'SigLIP 2 + FAISS | {language}', font=font(34), fill='#19334b')
    draw.text((45,85), query, font=font(29), fill='#202934')
    draw.text((45,135), f'Flickr8k: 300 images | Top {len(hits)} | Cosine similarity (not probability)', font=font(20), fill='#596574')
    for col, hit in enumerate(hits):
        x = 35 + col * 353
        draw.rounded_rectangle((x,190,x+335,730),radius=16,fill='white')
        with Image.open(BASE / hit['path']) as original:
            thumbnail = ImageOps.contain(original.convert('RGB'), (315,280))
        canvas.paste(thumbnail,(x+10+(315-thumbnail.width)//2,210+(280-thumbnail.height)//2))
        draw.text((x+15,515),f"#{hit['rank']}   {hit['score']:.4f}",font=font(26),fill='#19334b')
        for line_no, line in enumerate(textwrap.wrap(hit['captions'][0],width=29)[:6]):
            draw.text((x+15,562+line_no*24),line,font=font(18),fill='#303b49')
    draw.text((45,751),'Captions are shown for inspection only; the index contains image embeddings.',font=font(19),fill='#596574')
    canvas.save(RESULTS / filename)
    return canvas

# %% evaluation
def evaluate(index, records, processor, model):
    captions = [caption for r in records for caption in r['captions']]
    truth = np.repeat(np.arange(len(records)),5)
    vectors = embed_texts(captions,processor,model)
    scores, found = index.search(vectors,10)
    metrics = {f'Recall@{k}': float(np.mean(np.any(found[:,:k] == truth[:,None],axis=1))) for k in [1,5,10]}
    # 모든 300장 x 영어 캡션 5개 = 1,500개 질의. 부분집합 내 retrieval 평가.
    details = [{'image_id': records[int(t)]['id'], 'caption': c,
                'top10_ids': [records[int(j)]['id'] for j in indices],
                'correct_rank': next((rank+1 for rank,j in enumerate(indices) if j==t),None)}
               for t,c,indices in zip(truth,captions,found)]
    (RESULTS/'evaluation_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    report = {'dataset':DATASET,'revision':REVISION,'split':'test','seed':SEED,'images':len(records),
              'queries':len(captions),'model':MODEL_ID,'device':DEVICE,'dimension':index.d,
              'index':'IndexFlatIP (L2-normalized)','metrics':metrics,
              'scope':'English captions, 300-image candidate set. No training/fine-tuning. Not a full Flickr8k benchmark.',
              'versions':{'python':platform.python_version(),'torch':torch.__version__,'faiss':faiss.__version__}}
    (RESULTS/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report
