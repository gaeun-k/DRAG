"""Check saved submission artifacts without downloading or loading an AI model."""
import json
from pathlib import Path
import faiss
import numpy as np
import nbformat

base=Path(__file__).resolve().parent
result=base/'results'
records=json.loads((result/'manifest.json').read_text(encoding='utf-8'))
index=faiss.deserialize_index(np.frombuffer((result/'images.faiss').read_bytes(),dtype=np.uint8).copy())
assert len(records)==index.ntotal==300
assert len({r['id'] for r in records})==300
vectors=index.reconstruct_n(0,index.ntotal)
assert np.isfinite(vectors).all()
assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
metrics=json.loads((result/'metrics.json').read_text(encoding='utf-8'))
details=json.loads((result/'evaluation_details.json').read_text(encoding='utf-8'))
assert len(details)==1500
for k in [1,5,10]:
    recalculated=sum(d['image_id'] in d['top10_ids'][:k] for d in details)/len(details)
    assert abs(recalculated-metrics['metrics'][f'Recall@{k}'])<1e-12
nb=nbformat.read(base/'SigLIP_FAISS_실습.ipynb',as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type=='code':
        assert cell.execution_count is not None
        assert not any(o.output_type=='error' for o in cell.outputs)
assert len(list(result.glob('query_*.png')))==4
print('PASS: 300 vectors, unit norms, 1500 evaluation rows, recalculated Recall@K, executed notebook, four PNGs')
