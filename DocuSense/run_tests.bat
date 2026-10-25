@echo off
cd /d "%~dp0"
echo =======================================================
echo Executing DocuSense Automated Test Suite (pytest)...
echo Course: NAT820S - NLP and Applications (NUST Namibia)
echo =======================================================
pytest -v
pause
