import csv
import os

csv_path = r'C:\Users\User\Desktop\Projeto_Calculo_Numerico_GeoIP\logs_pings\historico_pings.csv'
if not os.path.exists(csv_path):
    print("CSV não encontrado!")
    exit(1)

with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Total de medições salvas no CSV: {len(rows)}")

timestamps = sorted(list(set(r['timestamp'] for r in rows)))
print(f"Total de rodadas de coleta realizadas: {len(timestamps)}")
print(f"Primeira medição registrada: {timestamps[0]}")
print(f"Última medição registrada   : {timestamps[-1]}")

dias = sorted(list(set(r['timestamp'].split()[0] for r in rows)))
print("\n--- Histórico por dia ---")
for dia in dias:
    rows_dia = [r for r in rows if r['timestamp'].startswith(dia)]
    rodadas_dia = len(set(r['timestamp'] for r in rows_dia))
    print(f"Data {dia}: {len(rows_dia)} pings registrados em {rodadas_dia} rodadas")

print(f"\n--- Detalhes da última rodada realizada ({timestamps[-1]}) ---")
ult_rows = [r for r in rows if r['timestamp'] == timestamps[-1]]
for r in ult_rows:
    nome = r.get('server_name', '')
    min_rtt = r.get('min_rtt_ms', '')
    avg_rtt = r.get('avg_rtt_ms', '')
    rec = r.get('received', '')
    sent = r.get('sent', '')
    if min_rtt:
        print(f"  [OK] {nome:30s} -> min={min_rtt}ms, media={avg_rtt}ms ({rec}/{sent})")
    else:
        print(f"  [--] {nome:30s} -> SEM RESPOSTA / TIMEOUT")
