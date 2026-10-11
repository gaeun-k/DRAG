from pathlib import Path
import nbformat
from nbclient import NotebookClient
p=Path(__file__).parent
nb=nbformat.read(p/'SigLIP_FAISS_실습.ipynb',as_version=4)
client=NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(p.resolve())}}, on_cell_start=lambda cell,cell_index: print(f'Cell {cell_index+1}/{len(nb.cells)}',flush=True))
try:
 client.execute()
finally:
 nbformat.write(nb,p/'SigLIP_FAISS_실습.ipynb')
print('Notebook execution complete',flush=True)
