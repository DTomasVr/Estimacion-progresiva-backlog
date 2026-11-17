#!/usr/bin/env bash
# Cuando acabe la pasada normal, repite en profundidad: con token hay presupuesto
# de sobra (5000/h) y las paginas sin cambios salen gratis por ETag.
set -u
cd "/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
cuenta() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object CommandLine -like '*50_scraper_github*' | Measure-Object).Count" 2>/dev/null | tr -d ' \r'; }
while [ "$(cuenta)" != "0" ]; do sleep 30; done
echo "[$(date '+%F %T')] pasada profunda: 25 paginas por repo" >> logs/50_scraper.log
PBE_PAGINAS=25 "$PY" src/50_scraper_github.py >> logs/50_scraper.log 2>&1
