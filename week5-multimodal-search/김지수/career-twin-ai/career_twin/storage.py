"""개인 문서 저장. 공개 합성 데이터는 이 저장소에 넣지 않습니다."""
from pathlib import Path
import hashlib
import json
import sqlite3
from datetime import datetime, timezone


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.root / "uploads"
        self.raw_dir.mkdir(exist_ok=True)
        self.database = self.root / "documents.sqlite3"
        with self.connect() as db:
            db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT, kind TEXT, sha256 TEXT,
                uploaded_at TEXT, path TEXT, warnings TEXT);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT, page INTEGER, section TEXT, position INTEGER);
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
                model TEXT, vector TEXT, PRIMARY KEY(chunk_id, model));
            """)

    def connect(self):
        db = sqlite3.connect(self.database)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def add(self, filename, content, kind, extracted, warnings=()):
        if kind not in {"personal", "paper"}:
            raise ValueError("개인 문서 또는 논문만 저장할 수 있습니다.")
        name = Path(filename.replace("\\", "/")).name
        sha = hashlib.sha256(content).hexdigest()
        doc_id = hashlib.sha256((kind + sha).encode()).hexdigest()[:24]
        suffix = Path(name).suffix.lower()
        path = self.raw_dir / (doc_id + suffix)
        with self.connect() as db:
            if db.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone():
                return doc_id
            path.write_bytes(content)
            try:
                db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", (
                    doc_id, name, kind, sha, datetime.now(timezone.utc).isoformat(),
                    str(path), json.dumps(list(warnings), ensure_ascii=False)))
                for i, item in enumerate(extracted):
                    db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)", (
                        f"{doc_id}:{i}", doc_id, item["text"], item.get("page"),
                        item.get("section", ""), i))
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return doc_id

    def documents(self, kind=None):
        with self.connect() as db:
            sql = "SELECT * FROM documents"
            return [dict(r) for r in db.execute(sql + (" WHERE kind=?" if kind else ""),
                                               (kind,) if kind else ())]

    def chunks(self, kind):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""
                SELECT c.*, d.name, d.kind, d.path FROM chunks c
                JOIN documents d ON c.document_id=d.id WHERE d.kind=? ORDER BY c.id
            """, (kind,))]

    def delete(self, doc_id):
        with self.connect() as db:
            row = db.execute("SELECT path FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row:
                db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        if row:
            Path(row["path"]).unlink(missing_ok=True)

    def get_vectors(self, model):
        with self.connect() as db:
            return {r["chunk_id"]: json.loads(r["vector"]) for r in
                    db.execute("SELECT * FROM vectors WHERE model=?", (model,))}

    def save_vectors(self, model, items):
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?)", [
                (chunk_id, model, json.dumps(vector)) for chunk_id, vector in items])
