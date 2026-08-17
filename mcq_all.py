#!/usr/bin/env python3
"""Entry point: python mcq_all.py --llm huatuo

Writes `questions.json` beside every document's artifacts.
"""
from medkg.mcq_corpus import main

if __name__ == "__main__":
    raise SystemExit(main())
