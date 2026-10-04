@echo off
cd /d
python main.py --serve --port 8112
python main.py --prefix /apb %*