"""Инициализация памяти после клонирования git-репозитория.

Проверяет и чинит всё, что не лежит под контролем версий: Python-venv, пакет fastembed,
ONNX-модель эмбеддингов, индекс memory.db. Идемпотентен: повторный запуск ничего лишнего
не делает.

Использование:
    python mem0-like\\init.py            # проверить и починить всё (нужен интернет, если чего-то нет)
    python mem0-like\\init.py --check    # только диагностика: ничего не создаёт и не качает
    python mem0-like\\init.py --force    # пересоздать индекс и перекачать модель заново

Всё скачанное и производное (.venv, .models, memory.db) намеренно вне git — этот скрипт
восстанавливает их из memory.md + сети.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(BASE, ".venv")
VENV_PY = os.path.join(VENV, "Scripts", "python.exe")
MODELS = os.path.join(BASE, ".models")
PIP_CACHE = os.path.join(BASE, ".pip-cache")
MEMORY_MD = os.path.join(BASE, "memory.md")
SEARCH_PY = os.path.join(BASE, "search.py")
DB = os.path.join(BASE, "memory.db")
REPO_DIR = os.path.join(
    MODELS, "models--Qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_ONNX = 100 * 1024 * 1024
MIN_TOKENIZER = 1 * 1024 * 1024


def base_python():
    """Интерпретатор, которым можно создать venv (не сам venv-питон)."""
    cfg = os.path.join(VENV, "pyvenv.cfg")
    if os.path.exists(cfg):
        for line in open(cfg, encoding="utf-8", errors="replace"):
            if line.strip().startswith("home"):
                home = line.split("=", 1)[1].strip()
                exe = os.path.join(home, "python.exe")
                if os.path.exists(exe):
                    return exe
    if os.path.normcase(sys.executable).startswith(os.path.normcase(VENV)):
        return None
    return sys.executable


def run(cmd, timeout=600):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=BASE
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def probe_embed(vpy, timeout=600):
    """Живой тест: venv + модель реально дают вектор. Возвращает (ok, сообщение)."""
    snippet = (
        "import os\n"
        "os.environ.setdefault('FASTEMBED_CACHE_PATH', %r)\n"
        "from fastembed import TextEmbedding\n"
        "m = TextEmbedding(model_name=%r)\n"
        "v = list(m.embed(['тест']))\n"
        "print(len(v[0]))\n"
    % (MODELS, MODEL_NAME))
    code, out, err = run([vpy, "-c", snippet], timeout=timeout)
    if code != 0:
        return False, (err or out).strip()[:300]
    return True, out.strip()[:60]


def model_files():
    """Файлы модели в кэше: (onnx, tokenizer, config) или None."""
    snaps = glob.glob(os.path.join(REPO_DIR, "snapshots", "*", "*"))
    onnx = tokenizer = config = None
    for p in snaps:
        name = os.path.basename(p)
        if name == "model_optimized.onnx":
            onnx = p
        elif name == "tokenizer.json":
            tokenizer = p
        elif name == "config.json":
            config = p
    if onnx and tokenizer and config:
        return onnx, tokenizer, config
    return None


def model_ok():
    """Наличие и целость файлов модели (размеры как грубая проверка целости)."""
    files = model_files()
    if files is None:
        return False, "файлы модели не найдены"
    onnx, tokenizer, config = files
    if os.path.getsize(onnx) < MIN_ONNX:
        return False, "model_optimized.onnx меньше ожидаемого (~%d МБ)" % (MIN_ONNX // 1024 // 1024)
    if os.path.getsize(tokenizer) < MIN_TOKENIZER:
        return False, "tokenizer.json меньше ожидаемого"
    try:
        json.load(open(config, encoding="utf-8"))
    except Exception as exc:
        return False, "config.json повреждён: %s" % exc
    return True, "файлы на месте"


def ensure_venv(check_only):
    if os.path.exists(VENV_PY):
        return True, "venv есть"
    if check_only:
        return False, "venv отсутствует (--check не создаёт)"
    exe = base_python()
    if not exe:
        return False, "не найден системный Python для создания venv"
    code, out, err = run([exe, "-m", "venv", VENV], timeout=300)
    if code != 0 or not os.path.exists(VENV_PY):
        return False, "venv не создан: %s" % (err or out).strip()[:200]
    return True, "venv создан"


def ensure_fastembed(vpy, check_only):
    code, out, err = run([vpy, "-c", "import fastembed"], timeout=120)
    if code == 0:
        return True, "fastembed установлен"
    if check_only:
        return False, "fastembed не установлен (--check не ставит)"
    os.makedirs(PIP_CACHE, exist_ok=True)
    code, out, err = run(
        [vpy, "-m", "pip", "install", "--cache-dir", PIP_CACHE, "fastembed"],
        timeout=900,
    )
    if code != 0:
        return False, "pip install не удался: %s" % (err or out).strip()[:250]
    code, out, err = run([vpy, "-c", "import fastembed"], timeout=120)
    if code != 0:
        return False, "fastembed после установки не импортируется"
    shutil.rmtree(PIP_CACHE, ignore_errors=True)
    return True, "fastembed установлен (пакеты скачаны)"


def ensure_model(vpy, check_only, force):
    ok, why = model_ok()
    if force and ok:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        ok, why = False, "перекачивание по --force"
    if ok:
        live, msg = probe_embed(vpy)
        if live:
            return True, "модель рабочая (размер вектора %s)" % msg
        why = "файлы есть, но модель не загружается: %s" % msg
    if check_only:
        return False, why + " (--check не качает)"
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    os.makedirs(MODELS, exist_ok=True)
    live, msg = probe_embed(vpy, timeout=900)
    if not live:
        return False, "загрузка модели не удалась: %s" % msg
    return True, "модель скачана и работает (%s)" % msg


def ensure_index(check_only, force):
    if not os.path.exists(MEMORY_MD):
        return False, "memory.md отсутствует"
    if check_only:
        sys.path.insert(0, BASE)
        import search

        con = search.connect()
        docs = search.read_entries()
        stored = con.execute("SELECT count(*) FROM doc").fetchone()[0]
        need = stored != len(docs) or force
        con.close()
        if need:
            return False, "индекс требует перестроения (%d документов, %d в индексе)" % (
                len(docs),
                stored,
            )
        return True, "индекс актуален (%d документов)" % stored
    sys.path.insert(0, BASE)
    import search

    con = search.connect()
    n, rebuilt = search.rebuild(con, force=force)
    con.close()
    return True, "индекс %s (%d документов)" % ("перестроен" if rebuilt else "актуален", n)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    check_only = "--check" in sys.argv
    force = "--force" in sys.argv
    results = []

    results.append(("memory.md", os.path.exists(MEMORY_MD), "есть" if os.path.exists(MEMORY_MD) else "нет"))
    results.append(
        ("search.py", os.path.exists(SEARCH_PY), "есть" if os.path.exists(SEARCH_PY) else "нет")
    )
    results.append(("git-репозиторий", os.path.exists(os.path.join(BASE, ".git")), "есть" if os.path.exists(os.path.join(BASE, ".git")) else "нет (.git не найден)"))

    venv_ok, venv_msg = ensure_venv(check_only)
    results.append(("venv (.venv)", venv_ok, venv_msg))

    if venv_ok and os.path.exists(VENV_PY):
        fe_ok, fe_msg = ensure_fastembed(VENV_PY, check_only)
        results.append(("пакет fastembed", fe_ok, fe_msg))
        if fe_ok:
            m_ok, m_msg = ensure_model(VENV_PY, check_only, force)
            results.append(("модель (.models)", m_ok, m_msg))
            idx_ok, idx_msg = ensure_index(check_only, force)
            results.append(("индекс (memory.db)", idx_ok, idx_msg))
        else:
            results.append(("модель (.models)", False, "пропуск: нет fastembed"))
            idx_ok, idx_msg = ensure_index(check_only, force)
            results.append(("индекс (memory.db)", idx_ok, idx_msg))
    else:
        results.append(("пакет fastembed", False, "пропуск: нет venv"))
        results.append(("модель (.models)", False, "пропуск: нет venv"))
        idx_ok, idx_msg = ensure_index(check_only, force)
        results.append(("индекс (memory.db)", idx_ok, idx_msg))

    print("Инициализация памяти: %s" % BASE)
    for name, ok, msg in results:
        print("%-20s %s  %s" % (name, "OK " if ok else "FAIL", msg))

    all_ok = all(ok for _, ok, _ in results)
    if all_ok:
        print("\nВсё готово. Поиск:")
        print('  %s "%s" -k 5' % (os.path.join(BASE, "search.py"), "ток по +12V"))
        print("  запуск из workspace: mem0-like\\.venv\\Scripts\\python.exe mem0-like\\search.py \"запрос\" -k 5")
        return 0
    print("\nЕсть проблемы — см. FAIL выше. Повторите без --check, чтобы починить.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
