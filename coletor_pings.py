#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coletor de Pings para o Projeto de Cálculo Numérico (Fase 7)
Salva medições brutas e resumo tabular para a estimativa de posição via Gauss-Newton.
Compatível com Windows (CP850 / CP1252 / UTF-8) forçando IPv4 e timeout rápido.
"""

import os
import sys
import re
import csv
import time
import subprocess
from datetime import datetime

# Garante suporte a caracteres no console Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVERS_CSV = os.path.join(SCRIPT_DIR, "servers.csv")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs_pings")
HISTORICO_CSV = os.path.join(LOGS_DIR, "historico_pings.csv")

os.makedirs(LOGS_DIR, exist_ok=True)

def carregar_servidores():
    servidores = []
    if not os.path.exists(SERVERS_CSV):
        print(f"[-] Arquivo {SERVERS_CSV} nao encontrado!")
        return servidores
    with open(SERVERS_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip().lower() != "ignorado":
                servidores.append(row)
    return servidores

def extrair_estatisticas_ping_windows(stdout_text):
    min_rtt, avg_rtt, max_rtt = None, None, None
    sent, received = 0, 0

    # Busca valores de resumo (Mínimo / Máximo / Média) individualmente
    min_m = re.search(r'(?:M[ií]nimo|Minimum)\s*=\s*(\d+(?:\.\d+)?)\s*ms', stdout_text, re.IGNORECASE)
    max_m = re.search(r'(?:M[aá]ximo|Maximum)\s*=\s*(\d+(?:\.\d+)?)\s*ms', stdout_text, re.IGNORECASE)
    avg_m = re.search(r'(?:M[eé]dia|Average)\s*=\s*(\d+(?:\.\d+)?)\s*ms', stdout_text, re.IGNORECASE)

    if min_m and max_m and avg_m:
        min_rtt = float(min_m.group(1))
        max_rtt = float(max_m.group(1))
        avg_rtt = float(avg_m.group(1))

    # Fallback: extrai de linhas individuais (tempo=XXms ou time=XXms)
    tempos = [float(x) for x in re.findall(r'(?:tempo|time)[=<](\d+(?:\.\d+)?)\s*ms', stdout_text, re.IGNORECASE)]
    if tempos and min_rtt is None:
        min_rtt = min(tempos)
        max_rtt = max(tempos)
        avg_rtt = sum(tempos) / len(tempos)

    # Pacotes enviados / recebidos
    pacotes_match = re.search(
        r'(?:Enviados|Sent)\s*=\s*(\d+).*?(?:Recebidos|Received)\s*=\s*(\d+)',
        stdout_text,
        re.IGNORECASE
    )
    if pacotes_match:
        sent = int(pacotes_match.group(1))
        received = int(pacotes_match.group(2))
    else:
        sent = 5
        received = len(tempos)

    return min_rtt, avg_rtt, max_rtt, sent, received

def executar_rodada_pings(num_pacotes=5):
    servidores = carregar_servidores()
    if not servidores:
        print("[-] Nenhum servidor carregado.")
        return

    agora = datetime.now()
    timestamp_str = agora.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = agora.strftime("%Y%m%d_%H%M%S")
    arquivo_raw = os.path.join(LOGS_DIR, f"ping_raw_{timestamp_file}.txt")

    print(f"\n=======================================================")
    print(f"[*] Iniciando rodada de ping: {timestamp_str}")
    print(f"[*] Servidores a testar: {len(servidores)}")
    print(f"[*] Destino dos logs: {LOGS_DIR}")
    print(f"=======================================================\n")

    registros_rodada = []
    conteudo_bruto = []

    for s in servidores:
        nome = s.get("name")
        host = s.get("host")
        lat = s.get("lat")
        lon = s.get("lon")

        print(f"[*] Pingando {nome} ({host})... ", end="", flush=True)

        # -4: Forca IPv4 (evita loops IPv6 em universidades)
        # -n: Numero de pacotes
        # -w 1000: Timeout de 1 segundo por pacote
        cmd = ["ping", "-4", "-n", str(num_pacotes), "-w", "1000", host]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=False, timeout=20)
            try:
                saida = proc.stdout.decode('cp850')
            except Exception:
                saida = proc.stdout.decode('utf-8', errors='replace')

            min_rtt, avg_rtt, max_rtt, sent, rec = extrair_estatisticas_ping_windows(saida)

            conteudo_bruto.append(f"=== {nome} ({host}) [{timestamp_str}] ===\n{saida}\n\n")

            if min_rtt is not None and rec > 0:
                print(f"OK! min={min_rtt:.1f}ms, media={avg_rtt:.1f}ms, max={max_rtt:.1f}ms (recebidos {rec}/{sent})")
            else:
                print(f"SEM RESPOSTA / TIMEOUT")

            registros_rodada.append({
                "timestamp": timestamp_str,
                "server_name": nome,
                "host": host,
                "lat": lat,
                "lon": lon,
                "sent": sent,
                "received": rec,
                "min_rtt_ms": f"{min_rtt:.2f}" if min_rtt is not None else "",
                "avg_rtt_ms": f"{avg_rtt:.2f}" if avg_rtt is not None else "",
                "max_rtt_ms": f"{max_rtt:.2f}" if max_rtt is not None else ""
            })

        except subprocess.TimeoutExpired:
            print("[TIMEOUT 20s]")
        except Exception as e:
            print(f"[ERRO: {e}]")

    # Salva log bruto da rodada
    with open(arquivo_raw, "w", encoding="utf-8") as f:
        f.writelines(conteudo_bruto)

    # Atualiza historico_pings.csv cumulativo
    csv_existe = os.path.exists(HISTORICO_CSV)
    campos = ["timestamp", "server_name", "host", "lat", "lon", "sent", "received", "min_rtt_ms", "avg_rtt_ms", "max_rtt_ms"]

    with open(HISTORICO_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if not csv_existe:
            writer.writeheader()
        writer.writerows(registros_rodada)

    print(f"\n[+] Rodada concluida com sucesso!")
    print(f"    -> Log bruto salvo em: {arquivo_raw}")
    print(f"    -> Tabela acumulada em: {HISTORICO_CSV}\n")

if __name__ == "__main__":
    if "--loop" in sys.argv:
        idx = sys.argv.index("--loop")
        intervalo = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 3600
        print(f"[*] Modo continuo ativado. Intervalo entre rodadas: {intervalo} segundos ({intervalo/60:.1f} min).")
        print("[*] Pressione Ctrl+C para interromper a qualquer momento.\n")
        try:
            while True:
                executar_rodada_pings()
                print(f"[~] Aguardando proxima rodada em {intervalo}s...")
                time.sleep(intervalo)
        except KeyboardInterrupt:
            print("\n[!] Coleta interrompida pelo usuario.")
    else:
        executar_rodada_pings()
