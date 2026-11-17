@echo off
title Coletor de Pings Continuo (a cada 10 minutos)
cd /d "%~dp0"
echo ==============================================================
echo  COLETOR AUTOMATICO DE PINGS - PROJETO CALCULO NUMERICO
echo  Intervalo otimizado: a cada 10 minutos (600 segundos).
echo  Pode minimizar esta janela e deixar no fundo do computador.
echo  Para interromper a coleta a qualquer momento, feche a janela.
echo ==============================================================
echo.
python coletor_pings.py --loop 600
pause
