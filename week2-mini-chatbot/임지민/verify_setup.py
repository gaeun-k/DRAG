"""Offline installation smoke check; --models additionally checks local Ollama."""
import argparse
from pathlib import Path
from advanced_rag import Options, StudyRag
from ingest import load_file
from local_ai import model_status
from runtime_policy import MAX_FILE_MB, require_allowed_mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    docs = load_file('study_wiki.md', (root / 'data/study_wiki.md').read_bytes())
    bot = StudyRag(docs, Options(mode='demo', k=2))
    answer = bot.ask('과제 제출물은 무엇인가요?')
    assert answer['sources'] and answer['answer'], 'No retrieval result'
    assert MAX_FILE_MB == 30
    try:
        require_allowed_mode('openai')
    except ValueError:
        pass
    else:
        raise AssertionError('Paid mode must be blocked')
    if args.models and not model_status():
        raise SystemExit('Local models missing. Run setup_local_ai.ps1.')
    print('PASS: imports, sample loading, FAISS retrieval, 30MB limit, free-only policy')
    if args.models:
        print('PASS: local chat and embedding models installed')

if __name__ == '__main__':
    main()
