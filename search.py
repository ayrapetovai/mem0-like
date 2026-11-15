r"""Гибридный поиск по памяти: SQLite FTS5 (точные токены) + векторы (смысл).

Использование:
    python mem0-like\\search.py "ток по +12V"
    python mem0-like\\search.py "кулер" -k 8
    python mem0-like\\search.py "враг" --topic game-design
    python mem0-like\\search.py "короб" --no-embed     # только FTS
    python mem0-like\\search.py --add "текст" --topic flow
    python mem0-like\\search.py --reindex
    python mem0-like\\search.py --stats

Источник: mem0-like\\memory\*.md — по файлу на тему. Каждая строка "- ..." это отдельный
документ; "#"/"##" задают заголовок раздела, он входит в текст документа.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "memory")
LEGACY_MD = os.path.join(BASE, "memory.md")
DB = os.path.join(BASE, "memory.db")
MODELS = os.path.join(BASE, ".models")
AUTO = "## Поток решений"
DEFAULT_TOPIC = "flow"
TOPIC_SPAN = 100000
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def doc_files():
    """Файлы тем: memory\\*.md; если папки нет — legacy memory.md как тема misc."""
    if os.path.isdir(DOCS_DIR):
        names = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".md"))
        return [(os.path.splitext(f)[0], os.path.join(DOCS_DIR, f)) for f in names]
    if os.path.exists(LEGACY_MD):
        return [("misc", LEGACY_MD)]
    return []


def parse_file(path):
    """(- bullet) строки файла → список текстов документов в порядке файла."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    docs = []
    heading = ""
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            heading = s.lstrip("#").strip()
            continue
        if s.startswith("- "):
            docs.append("%s :: %s" % (heading, s[2:]))
    return docs


def read_entries():
    """[(topic, num, text)] по всем файлам темы; num — позиция bullet в файле."""
    out = []
    for topic, path in doc_files():
        for num, text in enumerate(parse_file(path), start=1):
            out.append((topic, num, text))
    return out


def connect():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    row = con.execute("SELECT sql FROM sqlite_master WHERE name = 'doc'").fetchone()
    if row is None or "topic" not in (row[0] or "").lower():
        con.execute("DROP TABLE IF EXISTS doc")
        con.execute("DROP TABLE IF EXISTS vec")
        con.execute("DELETE FROM meta")
    con.execute(
        "CREATE TABLE IF NOT EXISTS doc ("
        "id INTEGER PRIMARY KEY, topic TEXT, num INTEGER, text TEXT)"
    )
    con.execute("CREATE TABLE IF NOT EXISTS vec (id INTEGER PRIMARY KEY, emb TEXT)")
    row = con.execute("SELECT sql FROM sqlite_master WHERE name = 'fts'").fetchone()
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


def topic_base(con, topic):
    """Стабильный диапазон id темы: номер темы закрепляется в meta навсегда."""
    stored = meta_get(con, "topic:" + topic)
    if stored is not None:
        return int(stored) * TOPIC_SPAN
    used = [int(v) for _, v in con.execute(
        "SELECT k, v FROM meta WHERE k LIKE 'topic:%'").fetchall()]
    index = max(used, default=0) + 1
    meta_set(con, "topic:" + topic, str(index))
    return index * TOPIC_SPAN


def embed(texts):
    """Векторы через fastembed; None, если библиотека не установлена."""
    if not texts:
        return []
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    model = TextEmbedding(model_name=MODEL_NAME)
    return [[float(x) for x in v] for v in model.embed(texts)]


def purge_topic(con, topic):
    con.execute(
        "DELETE FROM vec WHERE id IN (SELECT id FROM doc WHERE topic = ?)", (topic,)
    )
    con.execute(
        "DELETE FROM fts WHERE rowid IN (SELECT id FROM doc WHERE topic = ?)", (topic,)
    )
    con.execute("DELETE FROM doc WHERE topic = ?", (topic,))


def rebuild(con, force=False):
    """Индекс по файлам темы; эмбеддинги пересчитываются только для изменённых тем."""
    entries = read_entries()
    by_topic = {}
    for topic, num, text in entries:
        by_topic.setdefault(topic, []).append((num, text))
    stored = {t for (t,) in con.execute("SELECT DISTINCT topic FROM doc").fetchall()}
    changed = []
    for topic, items in sorted(by_topic.items()):
        digest = hashlib.sha256(
            "\n".join(t for _, t in items).encode("utf-8")
        ).hexdigest()
        if not force and meta_get(con, "hash:" + topic) == digest and topic in stored:
            continue
        base = topic_base(con, topic)
        purge_topic(con, topic)
        emb = embed([t for _, t in items])
        for i, (num, text) in enumerate(items):
            rid = base + num
            con.execute(
                "INSERT INTO doc (id, topic, num, text) VALUES (?, ?, ?, ?)",
                (rid, topic, num, text),
            )
            con.execute("INSERT INTO fts (rowid, text) VALUES (?, ?)", (rid, text))
            if emb is not None:
                con.execute(
                    "INSERT INTO vec (id, emb) VALUES (?, ?)", (rid, json.dumps(emb[i]))
                )
        meta_set(con, "hash:" + topic, digest)
        meta_set(con, "embedded", "1" if emb is not None else "0")
        changed.append(topic)
    for topic in sorted(stored - set(by_topic)):
        purge_topic(con, topic)
        meta_set(con, "hash:" + topic, "")
        changed.append(topic)
    con.commit()
    return len(entries), changed


def fts_search(con, query, limit, topic=None):
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        return []
    match = " OR ".join('"%s"' % t for t in tokens)
    sql = "SELECT f.rowid FROM fts f JOIN doc d ON d.id = f.rowid WHERE f MATCH ?"
    args = [match]
    if topic:
        sql += " AND d.topic = ?"
        args.append(topic)
    sql += " ORDER BY bm25(f) LIMIT ?"
    args.append(limit)
    try:
        return [r[0] for r in con.execute(sql, args).fetchall()]
    except sqlite3.OperationalError:
        return []


def vec_search(con, query, limit, topic=None):
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        return []
    sql = "SELECT v.id, v.emb FROM vec v JOIN doc d ON d.id = v.id"
    args = []
    if topic:
        sql += " WHERE d.topic = ?"
        args.append(topic)
    stored = con.execute(sql, args).fetchall()
    if not stored:
        return []
    q = embed([query])
    if not q:
        return []
    qv = q[0]
    norm = math.sqrt(sum(c * c for c in qv)) or 1.0
    scored = []
    for rid, raw in stored:
        v = json.loads(raw)
        dot = 0.0
        nn = 0.0
        for a, b in zip(qv, v):
            dot += a * b
            nn += b * b
        scored.append((dot / (norm * (math.sqrt(nn) or 1.0)), rid))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:limit]]


def fts_only_search(con, query, limit, topic=None):
    """Рейтинг без BM25: сколько токенов встречается в документе."""
    tokens = [t.lower() for t in TOKEN_RE.findall(query)]
    if not tokens:
        return []
    sql = "SELECT id, text FROM doc"
    args = []
    if topic:
        sql += " WHERE topic = ?"
        args.append(topic)
    scored = []
    for rid, text in con.execute(sql, args).fetchall():
        low = text.lower()
        hits = sum(1 for t in tokens if t in low)
        if hits:
            scored.append((hits, rid))
    scored.sort(reverse=True)
    return [rid for _, rid in scored[:limit]]


def search(con, query, k, use_embed, topic=None):
    want = 40 if topic else 20
    if use_embed:
        fts = fts_search(con, query, want, topic)
        vec = vec_search(con, query, want, topic)
    else:
        fts = fts_only_search(con, query, want, topic)
        vec = []
    if not fts and not vec:
        return []
    scores = {}
    for rank, rid in enumerate(fts, start=1):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (60 + rank)
    for rank, rid in enumerate(vec, start=1):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (60 + rank)
    out = []
    for rid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]:
        topic_, num, text = con.execute(
            "SELECT topic, num, text FROM doc WHERE id = ?", (rid,)
        ).fetchone()
        out.append((score, "%s:%d" % (topic_, num), text))
    return out


def known_topics(con):
    names = {os.path.splitext(p)[0] for _, p in doc_files()}
    names |= {t for (t,) in con.execute("SELECT DISTINCT topic FROM doc").fetchall()}
    names |= {k[len("topic:"):] for k, _ in con.execute(
        "SELECT k, v FROM meta WHERE k LIKE 'topic:%'").fetchall()}
    return sorted(names)


def resolve_topic(con, name, create):
    """Имя темы: точное имя файла, иначе уникальное совпадение подстрокой."""
    if not name:
        return DEFAULT_TOPIC
    names = known_topics(con)
    if name in names:
        return name
    hits = [t for t in names if name in t]
    if len(hits) == 1:
        return hits[0]
    if create and re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return name
    raise SystemExit(
        "тема %r не найдена; известные: %s" % (name, ", ".join(names) or "нет")
    )


def append_entry(text, topic):
    path = os.path.join(DOCS_DIR, topic + ".md")
    os.makedirs(DOCS_DIR, exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# %s\n\n%s\n\n" % (topic, AUTO)
    if AUTO not in content:
        content = content.rstrip() + "\n\n" + AUTO + "\n\n"
    if not content.endswith("\n"):
        content += "\n"
    content += "- " + text.strip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="Гибридный поиск по mem0-like\\memory\\*.md"
    )
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--topic", default=None, help="фильтр или тема для --add")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--add", default=None)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    con = connect()

    if a.add:
        topic = resolve_topic(con, a.topic, create=True)
        append_entry(a.add, topic)
        n, changed = rebuild(con)
        print("добавлено в %s: %d документов, перестроено тем: %d"
              % (topic, n, len(changed)))
        return 0

    n, changed = rebuild(con, force=a.reindex)

    if a.stats:
        counts = con.execute(
            "SELECT topic, COUNT(*) FROM doc GROUP BY topic ORDER BY topic"
        ).fetchall()
        print("документов: %d" % n)
        for topic, count in counts:
            print("  %-14s %d" % (topic, count))
        print("индекс: %s" % ("перестроен: " + ", ".join(changed)
                              if changed else "актуален"))
        print("векторы: %s" % (meta_get(con, "embedded") or "нет"))
        print("модель: %s" % MODEL_NAME)
        return 0

    topic = resolve_topic(con, a.topic, create=False) if a.topic else None

    if not a.query:
        print("нужен запрос или --add/--stats")
        return 1

    results = search(con, a.query, a.k, use_embed=not a.no_embed, topic=topic)
    if not results:
        print("совпадений нет")
        return 0
    for score, label, text in results:
        shown = text if len(text) <= 240 else text[:240] + "…"
        print("[%.4f] #%s :: %s" % (score, label, shown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
