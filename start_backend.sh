#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
uvicorn main:app --reload --port 8000
