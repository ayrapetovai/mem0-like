# Память проекта — указатель

Текст памяти разложен по темам в `mem0-like\memory\`; этот файл ничего не индексируется.

- `memory\hardware.md` — сборка Strata node: CPU, плата, RAM, кулер, короб, БП, сеть, диск, fan-control.
- `memory\game-design.md` — архитектура генеративной Rogue-игры с LLM: база фактов, ISA, канон, бой, мультиплеер, модель и рантайм.
- `memory\flow.md` — поток решений по workspace и окружению.

Поиск по всем темам сразу: `mem0-like\.venv\Scripts\python.exe mem0-like\search.py "запрос" -k 5`
Одна тема: `... search.py "запрос" --topic game-design`
Запись решения: `... search.py --add "текст" --topic flow`
