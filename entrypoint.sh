#!/usr/bin/env bash
set -e
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}
