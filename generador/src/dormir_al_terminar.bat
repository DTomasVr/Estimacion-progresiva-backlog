@echo off
REM Doble clic aqui para que la PC se duerma sola cuando Claude termine todo.
REM Espera la senal logs\LISTO.flag, comprueba que no quede nada calculando,
REM y entonces suspende (no hiberna). Cierra esta ventana para cancelar.
title Dormir al terminar - PBE
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dormir_al_terminar.ps1" %*
echo.
echo (si ves esto y la PC no se durmio, revisa el mensaje de arriba)
pause
