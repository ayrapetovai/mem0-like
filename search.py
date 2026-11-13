"""Гибридный поиск по памяти: SQLite FTS5 (точные токены) + векторы (смысл).

Использование:
    python mem0-like\\search.py "ток по +12V"
    python mem0-like\\search.py "кулер" -k 8
    python mem0-like\\search.py "короб" --no-embed     # только FTS
    python mem0-like\\search.py --add "новая запись"   # append в memory.md + переиндексация
    python mem0-like\\search.py --reindex
    python mem0-like\\search.py --stats

Индекс строится из memory.md: каждая строка вида "- ..." это отдельный документ.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(BASE, "memory.md")
DB = os.path.join(BASE, "memory.db")
MODELS = os.path.join(BASE, ".models")
AUTO = "## Поток решений"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
os.environ.setdefault("FASTEMBED_CACHE_PATH", MODELS)

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def read_entries():
    """(- bullet) строки memory.md → список документов "раздел :: текст"."""
    if not os.path.exists(MD):
        return []
    with open(MD, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    docs = []
    heading = ""
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            heading = s.lstrip("#").strip()
            continue
        if s.startswith("- "):
            docs.append(f"{heading} :: {s[2:]}")
    return docs


def connect():
    con = sqlite3.connect(DB)
    con.execute(
        "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS doc (id INTEGER PRIMARY KEY, text TEXT)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS vec (id INTEGER PRIMARY KEY, emb TEXT)"
    )
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'fts'"
    ).fetchone()
    if row is None or "fts5" not in (row[0] or "").lower():
        con.execute("DROP TABLE IF EXISTS fts")
        con.execute(
            "CREATE VIRTUAL TABLE fts USING fts5("
            "text, tokenize='unicode61 remove_diacritics 2')"
        )
    con.commit()
    return con


def meta_get(con, k):
    row = con.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
    return row[0] if row else None


def meta_set(con, k, v):
    con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, v))


def embed(texts):
    """Векторы через fastembed; None, если библиотека не установлена."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    model = TextEmbedding(model_name=MODEL_NAME)
    return [[float(x) for x in v] for v in model.embed(texts)]


def rebuild(con, force=False):
    docs = read_entries()
    digest = hashlib.sha256("\n".join(docs).encode("utf-8")).hexdigest()
    if not force and meta_get(con, "docs_hash") == digest:
        return len(docs), False
    con.execute("DELETE FROM doc")
    con.execute("DELETE FROM fts")
    con.execute("DELETE FROM vec")
    for i, text in enumerate(docs, start=1):
        con.execute("INSERT INTO doc (id, text) VALUES (?, ?)", (i, text))
        con.execute("INSERT INTO fts (rowid, text) VALUES (?, ?)", (i, text))
    vecs = embed(docs)
    if vecs is not None:
        for i, v in enumerate(vecs, start=1):
            con.execute(
                "INSERT INTO vec (id, emb) VALUES (?, ?)", (i, json.dumps(v))
            )
    meta_set(con, "docs_hash", digest)
    meta_set(con, "embedded", "1" if vecs is not None else "0")
    con.commit()
    return len(docs), True


def fts_search(con, query, limit):
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        return []
    match = " OR ".join('"%s"' % t for t in tokens)
    try:
        rows = con.execute(
            "SELECT rowid FROM fts WHERE fts MATCH ? "
            "ORDER BY bm25(fts) LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r[0] for r in rows]


def vec_search(con, query, limit):
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        return []
    stored = con.execute("SELECT id, emb FROM vec").fetchall()
    if not stored:
        return []
    q = embed([query])
    if not q:
        return []
    qv = q[0]
    norm = math_sqrt(sum(c * c for c in qv)) or 1.0
    scored = []
    for rid, raw in stored:
        v = json.loads(raw)
        dot = 0.0
        nn = 0.0
        for a, b in zip(qv, v):
            dot += a * b
            nn += b * b
        vn = math_sqrt(nn) or 1.0
        scored.append((dot / (norm * vn), rid))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:limit]]


def math_sqrt(x):
    return x ** 0.5


def fts_only_search(con, query, limit):
    """Рейтинг без BM25: сколько токенов встречается в документе."""
    tokens = [t.lower() for t in TOKEN_RE.findall(query)]
    if not tokens:
        return []
    rows = con.execute("SELECT id, text FROM doc").fetchall()
    scored = []
    for rid, text in rows:
        low = text.lower()
        hits = sum(1 for t in tokens if t in low)
        if hits:
            scored.append((hits, rid))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:limit]]


def search(con, query, k, use_embed):
    fts = fts_search(con, query, 20) if use_embed else []
    vec = vec_search(con, query, 20) if use_embed else []
    if not use_embed:
        fts = fts_only_search(con, query, 20)
    if not fts and not vec:
        return []
    scores = {}
    for rank, rid in enumerate(fts, start=1):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (60 + rank)
    for rank, rid in enumerate(vec, start=1):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out = []
    for rid, score in ranked:
        text = con.execute(
            "SELECT text FROM doc WHERE id = ?", (rid,)
        ).fetchone()[0]
        out.append((score, rid, text))
    return out


def append_entry(text):
    with open(MD, "r", encoding="utf-8") as f:
        content = f.read()
    if AUTO not in content:
        content = content.rstrip() + "\n\n" + AUTO + "\n\n"
    if not content.endswith("\n"):
        content += "\n"
    content += "- " + text.strip() + "\n"
    with open(MD, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Гибридный поиск по memory.md")
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--add", default=None)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    if a.add:
        append_entry(a.add)
    con = connect()
    n, changed = rebuild(con, force=a.reindex)

    if a.stats:
        print(f"документов: {n}")
        print(f"индекс: {'перестроен' if changed else 'актуален'}")
        print(f"векторы: {meta_get(con, 'embedded') or 'нет'}")
        print(f"модель: {MODEL_NAME}")
        return 0

    if not a.query:
        if a.add:
            print(f"записано, документов: {n}")
            return 0
        print("нужен запрос или --add/--stats")
        return 1

    results = search(con, a.query, a.k, use_embed=not a.no_embed)
    if not results:
        print("совпадений нет")
        return 0
    for score, rid, text in results:
        shown = text if len(text) <= 240 else text[:240] + "…"
        print(f"[{score:.4f}] #{rid} {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
