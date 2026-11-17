# SDD — Onde está esse IP?
**Geolocalização de máquinas por latência de rede como problema de mínimos quadrados não linear**

> Documento de Desenho de Software (Software Design Document) para guiar, passo a passo, a execução do projeto final de Cálculo Numérico.
> Integrantes: Gabriel Santana Albuquerque de Moraes (177927) · Luís Fernando Giaquinto Zólio (185241) · Lucas dos Santos Silva (176570) · Luís Gustavo Cardoso (185627)

---

## 1. Visão geral

### 1.1 Problema
Estimar a posição geográfica (latitude, longitude) de uma máquina usando apenas o RTT (ping) medido a partir de servidores de posição conhecida. É uma variante do GPS: o atraso fixo desconhecido `b` faz o papel do viés do relógio.

### 1.2 Pergunta de pesquisa
Como a **geometria** dos servidores (condicionamento da matriz G) e a **natureza da rede** (rotas não retas, latência inflada) afetam a precisão da posição estimada?

### 1.3 Critérios de sucesso (definição de "pronto")
- [ ] Gauss-Newton converge em dados sintéticos a partir de ≥ 3 chutes iniciais diferentes.
- [ ] Duas vias de solução (equações normais e QR) implementadas e comparadas em precisão simples (float32) e dupla (float64).
- [ ] Métrica de DOP implementada e correlacionada com o erro de Monte Carlo.
- [ ] Experimento de geometria boa vs. ruim com diferença de erro claramente demonstrada.
- [ ] Localização de São Paulo com dados reais, com erro observado comparado ao previsto.
- [ ] Discussão das limitações (assimetria do erro por inflação de latência).
- [ ] Relatório final + apresentação entregues até 04/12.

### 1.4 Fora de escopo
Geolocalização baseada em topologia/traceroute, bancos de GeoIP comerciais, localização de múltiplos alvos. Levenberg-Marquardt e restrições tipo CBG (Gueye et al.) entram apenas como **extensão opcional**.

---

## 2. Modelo matemático

### 2.1 Convenções
| Símbolo | Significado | Unidade |
|---|---|---|
| `x = (φ, λ)` | posição do alvo (lat, lon) | graus (rad internamente) |
| `sᵢ = (φᵢ, λᵢ)` | posição do servidor i, i = 1..m | graus |
| `tᵢ` | RTT medido ao servidor i | ms |
| `v` | velocidade efetiva | ≈ 100 km por ms de RTT |
| `b` | atraso fixo desconhecido | ms |
| `R` | raio da Terra | 6371 km |
| `dᵢ(x)` | distância de círculo máximo (haversine) | km |

### 2.2 Modelo direto
```
tᵢ = dᵢ(x) / v + b + εᵢ
```
**Recomendação de implementação:** trabalhar em **km**. Definir `β = v·b` (atraso fixo em km) e resíduos

```
rᵢ(θ) = dᵢ(φ, λ) + β − v·tᵢ ,    θ = (φ, λ, β)
```
Isso deixa as três colunas de G na mesma escala e melhora o condicionamento numérico (documentar essa escolha no relatório).

### 2.3 Problema de mínimos quadrados não linear
```
min_θ  ½ ‖r(θ)‖₂²
```
Precisa de m ≥ 3 servidores (3 incógnitas); usar m ≥ 6 em geral.

### 2.4 Gauss-Newton
A cada iteração k:
1. Calcular `r(θₖ)` e a Jacobiana `G = ∂r/∂θ` (m × 3).
2. Resolver `min_δ ‖G δ + r‖₂` (via equações normais **ou** QR).
3. Atualizar `θₖ₊₁ = θₖ + δ`.

**Jacobiana (linha i):**
```
G[i,:] = [ ∂dᵢ/∂φ , ∂dᵢ/∂λ , 1 ]
```
- Derive analiticamente a partir da fórmula de haversine (ou via vetores unitários 3D) e **valide com diferenças finitas** (teste unitário obrigatório).
- Cuidado com `dᵢ → 0` (derivada indefinida) e com os polos/antimeridiano (λ ±180°).

**Equivalência com GPS (útil para DOP):** no plano tangente local, a linha i de G é `[−uᵢ,leste, −uᵢ,norte, 1]` (ajustando sinal/escala), onde `uᵢ` é o vetor unitário do alvo ao servidor i.

### 2.5 Critério de parada (documentar no relatório)
Parar quando **qualquer** condição for satisfeita:
- `‖δₖ‖ < tol_x` (ex.: 1e-3 km) — passo pequeno;
- `|‖rₖ‖ − ‖rₖ₋₁‖| / ‖rₖ‖ < tol_f` — estagnação do resíduo;
- `k ≥ k_max` (ex.: 50) — falha por não convergência (registrar como tal);
- divergência: `‖rₖ‖` crescendo por 3 iterações seguidas.

### 2.6 Diluição de precisão (DOP)
```
Q = (GᵀG)⁻¹
GDOP = sqrt(trace(Q))                 # inclui b
PDOP = sqrt(Q₁₁ + Q₂₂)                # só posição
```
Relação esperada (σ = desvio padrão do ruído em km): `erro_RMS_posição ≈ σ · PDOP`.
Comparar com `κ₂(G) = σ_max(G) / σ_min(G)`.

### 2.7 Equações normais vs. QR
- **Normais:** resolver `GᵀG δ = −Gᵀr` (Cholesky). `κ(GᵀG) = κ(G)²`.
- **QR (Householder):** `G = QR`, resolver `R δ = −Qᵀr` por substituição regressiva.
- **Previsão teórica:** erro das normais ≈ `eps · κ²`; QR ≈ `eps · κ`. Em float32 (eps ≈ 1.2e-7) as normais perdem precisão por volta de κ ≈ 10³; em float64 (eps ≈ 2.2e-16), por volta de κ ≈ 10⁷–10⁸. **Verificar experimentalmente.**

---

## 3. Arquitetura do software

### 3.1 Stack
- **Python 3.11+**, `numpy`, `matplotlib`, `pandas`, `pytest`.
- Opcional: `cartopy` ou `geopandas` (mapas), `scipy` (**apenas para validar** suas implementações, não para substituí-las).
- A fatoração QR (Householder) e o Cholesky/substituições **devem ser implementados pelo grupo** (é o ponto do trabalho). `numpy.linalg.qr`/`lstsq` servem só como referência de teste.

### 3.2 Estrutura de pastas
```
projeto-geoip/
├── README.md
├── requirements.txt
├── src/
│   ├── geo.py            # haversine, conversões lat/lon, vetores unitários
│   ├── model.py          # residuo(θ), jacobiana(θ), conversão RTT→km
│   ├── solvers.py        # solve_normal(G, r, dtype), solve_qr(G, r, dtype), Householder, Cholesky
│   ├── gauss_newton.py   # laço principal, critério de parada, histórico
│   ├── dop.py            # GDOP, PDOP, cond(G)
│   ├── synthetic.py      # gerador de servidores, medidas com ruído, geometrias
│   ├── montecarlo.py     # experimentos de erro vs ruído/geometria
│   └── real_data.py      # leitura de medições reais, pré-processamento (min RTT)
├── data/
│   ├── servers.csv       # nome, lat, lon, fonte
│   └── pings_raw/        # medições brutas com timestamp
├── experiments/
│   ├── exp01_convergencia.py
│   ├── exp02_normais_vs_qr.py
│   ├── exp03_geometria_dop.py
│   ├── exp04_montecarlo.py
│   └── exp05_dados_reais.py
├── tests/
│   ├── test_geo.py
│   ├── test_jacobiana.py
│   ├── test_solvers.py
│   └── test_gauss_newton.py
├── results/              # figuras e tabelas geradas
└── report/               # relatório final (LaTeX/Word) e slides
```

### 3.3 Interfaces principais
```python
# geo.py
haversine(lat1, lon1, lat2, lon2) -> float   # km

# model.py
residuals(theta, servers, rtt_ms, v=100.0) -> np.ndarray      # (m,)
jacobian(theta, servers, v=100.0)          -> np.ndarray      # (m,3)

# solvers.py
solve_normal(G, r, dtype=np.float64) -> np.ndarray   # δ via Cholesky
solve_qr(G, r, dtype=np.float64)     -> np.ndarray   # δ via Householder

# gauss_newton.py
gauss_newton(servers, rtt_ms, theta0, solver="qr", dtype=np.float64,
             tol_x=1e-3, tol_f=1e-8, max_iter=50) -> Result
# Result: theta, converged(bool), n_iter, history(θₖ, ‖r‖, ‖δ‖), cond_G

# dop.py
dop_metrics(G) -> dict(gdop, pdop, cond2)
```

### 3.4 Decisões de projeto a registrar
1. Terra esférica (R = 6371 km) — erro de modelo ≤ ~0,3% frente ao elipsoide, desprezível frente ao erro de latência.
2. Unidades internas em km e radianos.
3. Reprodutibilidade: toda simulação usa `np.random.default_rng(seed)` com semente registrada.
4. Resultados de cada experimento salvos em `results/` (CSV + PNG) por script, sem passos manuais.

---

## 4. Plano de implementação passo a passo

> **Cronograma reestimado a partir de hoje (segunda, 26/10/2026) até a entrega em 04/12/2026** (~5,5 semanas úteis). Assume-se que nada do cronograma original foi concluído. Se algo já estiver pronto, marque os checklists e adiante a fase seguinte.
>
> **Feriados no período:** segunda 02/11 (Finados) e sexta 20/11 (Consciência Negra) — ambos já descontados. O domingo 15/11 (Proclamação da República) não afeta.
>
> **Mudança estratégica:** a coleta de dados reais (Fase 7) passa a rodar **em paralelo desde a primeira semana**, porque as medições precisam ser feitas em vários horários e dias. O rascunho do relatório (Fase 10) também começa antes, de forma incremental.

### Visão geral do novo cronograma

| Semana | Período | Fases | Marco de saída |
|---|---|---|---|
| 1 | 26/10–30/10 | 1, 2, início da 3, início da 7 | Repositório pronto; haversine/Jacobiana testadas; script de ping rodando em segundo plano |
| 2 | 02/11 (feriado)–06/11 | 3, início da 4 | Gauss-Newton com normais funcionando; Exp01 (convergência) concluído |
| 3 | 09/11–13/11 | 4, 5, fim da 7 | QR + Exp02 (normais vs. QR); DOP + Exp03 (geometrias); dados reais coletados e limpos |
| 4 | 16/11–19/11 (20/11 feriado) | 6, 8 (em paralelo) | Monte Carlo concluído; localização com dados reais rodando |
| 5 | 23/11–26/11 | 8 (fim), 9, 10 (rascunho) | **Congelamento do código em 26/11**; análise de limitações; relatório completo em rascunho |
| 6 | 27/11–04/12 | 10 (final), 11 | Relatório final em 01/12; revisão, slides e entrega em 04/12 |

**Marcos de controle:** (1) 30/10 — Jacobiana validada e coleta de pings iniciada; (2) 06/11 — Gauss-Newton convergindo em sintético; (3) 13/11 — Exp01, Exp02 e Exp03 prontos; (4) 19/11 — Monte Carlo pronto; (5) 26/11 — código congelado; (6) 01/12 — relatório final.

### Fase 1 — Revisão bibliográfica, setup e divisão de tarefas (26/10–27/10)
- [ ] Ler: Strang & Borre (cap. GPS), Trefethen & Bau (aulas sobre QR e condicionamento), Padmanabhan & Subramanian, Gueye et al.
- [ ] Definir responsáveis (sugestão):
  - **Pessoa A:** modelo + Gauss-Newton (`geo`, `model`, `gauss_newton`)
  - **Pessoa B:** solvers (Householder, Cholesky) e comparação float32/float64
  - **Pessoa C:** sintético, Monte Carlo, DOP
  - **Pessoa D:** dados reais, mapas, redação
  - Todos: revisão cruzada de código e relatório.
- [ ] Criar repositório Git, `README`, `requirements.txt`.
- [ ] Escolher a lista de servidores candidatos (para a Fase 7 começar já em 28/10).
- [ ] Revisão bibliográfica enxuta (2 dias): foco em Strang & Borre, Trefethen & Bau (QR e condicionamento) e Gueye et al.; o restante fica como leitura de apoio durante o semestre.

### Fase 2 — Modelagem (27/10–29/10)
*Responsável principal: Pessoa A.*

- [ ] Implementar `haversine` e testar (ex.: SP–Rio ≈ 360 km; SP–Lisboa ≈ 7.900 km).
- [ ] Implementar `residuals` e `jacobian` em km.
- [ ] **Teste:** Jacobiana analítica vs. diferenças finitas (erro relativo < 1e-6).
- [ ] Documentar a conversão RTT→distância (≈ 100 km/ms de RTT) e a interpretação de `b`.

### Fase 3 — Gauss-Newton com equações normais (29/10–06/11)
- [ ] `solve_normal` com Cholesky próprio.
- [ ] Laço de Gauss-Newton com histórico por iteração.
- [ ] Gerador sintético: 6–10 servidores no Brasil/mundo, alvo em São Paulo, `b` conhecido, ruído gaussiano.
- [ ] **Teste:** sem ruído, recupera `(φ, λ, β)` com erro ~1e-8.
- [ ] **Exp01 — convergência:** chutes iniciais (a) centro do Brasil, (b) ponto a 1.000 km da solução, (c) ponto distante (ex.: 3.000 km), (d) chute ruim (hemisfério trocado, para mostrar falha). Registrar nº de iterações, trajetória no mapa, ‖δ‖ × k (escala log).

### Fase 4 — Fatoração QR e precisão simples/dupla (04/11–11/11)
- [ ] `solve_qr` com Householder (e, se der tempo, Gram-Schmidt modificado para comparar).
- [ ] **Teste:** concordância com `numpy.linalg.lstsq` em float64.
- [ ] **Exp02 — normais vs. QR:**
  1. Construir G sintéticas com `κ₂` controlado (de 10⁰ a 10¹⁰), via SVD: `G = U Σ Vᵀ` com Σ escolhida.
  2. Fixar solução exata `δ*`, calcular `r = −G δ*` e resolver pelas duas vias em float32 e float64.
  3. Plotar erro relativo × κ em log-log (4 curvas: normais/QR × float32/float64).
  4. Identificar o κ em que as normais divergem do QR e comparar com `1/√eps`.
- [ ] Repetir com G reais (de geometrias de servidores) para conectar com o problema.

### Fase 5 — DOP e número de condição (10/11–13/11)
- [ ] Implementar `dop_metrics(G)`.
- [ ] Definir geometrias:
  - **Boa:** servidores espalhados em todas as direções ao redor do alvo (azimutes ~uniformes).
  - **Ruim:** todos na mesma direção (ex.: só Europa para alvo em SP), ou quase colineares.
  - **Intermediária:** Américas apenas, etc.
- [ ] **Exp03:** tabela geometria × (PDOP, GDOP, κ₂(G), erro). Gráfico PDOP × κ₂ (correlação).
- [ ] Mostrar que, com `b` incluído, a geometria "só Europa" torna a coluna de `b` quase colinear com as demais → κ alto.
- [ ] Gerar mapa com os servidores de cada geometria.

### Fase 6 — Monte Carlo (16/11–19/11)
- [ ] Para cada geometria e cada σ ∈ {0.5, 1, 2, 5, 10} ms (ou em km): rodar N ≥ 1000 realizações com `rng` fixo.
- [ ] Registrar erro de posição (km) por tentativa: média, RMS, percentis 50/95, taxa de não convergência.
- [ ] **Validar a hipótese:** `erro_RMS ≈ σ_km · PDOP` — plot de erro empírico × previsto (deve cair perto da reta y = x).
- [ ] Plot de erro × σ (deve ser ~linear) e erro × PDOP.
- [ ] Discutir desvios (não linearidade em geometria ruim, onde a linearização falha).
- [ ] Salvar CSV + figuras em `results/`.

### Fase 7 — Coleta de dados reais (28/10–13/11, em paralelo)
- [ ] Escolher ≥ 10–15 servidores públicos com localização **confiável e conhecida**, bem distribuídos: universidades (ex.: USP, UFRJ, Univ. Lisboa, MIT, Univ. Tóquio…), provedores/CDNs, pontos de troca de tráfego (IX.br), sondas RIPE Atlas.
- [ ] Criar `servers.csv` com nome, host/IP, lat, lon, fonte da localização.
- [ ] Script de coleta (`ping -c 50`) pronto até **28/10** e agendado (cron/Agendador de Tarefas) para rodar em vários horários (manhã, tarde, noite, madrugada) durante ~2 semanas, registrando timestamp.
- [ ] Limpeza e pré-processamento dos dados em 12/11–13/11 (não esperar o fim das coletas para escrever o código de leitura).
- [ ] **Pré-processamento:** usar o **mínimo** (ou percentil baixo) do RTT por servidor, pois o atraso por fila só infla.
- [ ] Remover/sinalizar servidores com RTT fisicamente impossível (distância > 100 km/ms · RTT) ou anycast (CDNs podem responder de um nó próximo e enganar o modelo; checar).
- [ ] Guardar dados brutos em `data/pings_raw/` para reprodutibilidade.

### Fase 8 — Localização com dados reais (17/11–24/11, em paralelo com a Fase 6)
- [ ] Pré-requisito: solvers da Fase 4 concluídos (11/11) e dados da Fase 7 limpos (13/11).
- [ ] Rodar Gauss-Newton (QR em float64) com os RTTs mínimos; alvo conhecido: sua máquina em São Paulo.
- [ ] Reportar: posição estimada, erro em km, `b` estimado, resíduos por servidor.
- [ ] Repetir com diferentes subconjuntos de servidores (boa vs. ruim geometria real) e diferentes horários.
- [ ] Análise de convergência com os chutes iniciais da Fase 3 e justificativa do critério de parada.
- [ ] Calcular PDOP/κ dos servidores reais usados e **comparar erro observado × erro previsto**.

### Fase 9 — Análise de erros e limitações (23/11–26/11)
- [ ] **Inflação de latência:** distância estimada ≥ distância real (RTT infla, nunca reduz). Mostrar nos resíduos que o erro é assimétrico (viés positivo), em contraste com o ruído gaussiano simétrico do modelo.
- [ ] Rotas indiretas: calcular a razão `RTT_observado·v / distância_real` (fator de circuito) por servidor.
- [ ] Impacto do viés no `b` estimado (absorve parte da inflação média).
- [ ] Teste de sensibilidade: adicionar nas simulações ruído **assimétrico** (exponencial/lognormal positivo) e comparar com o gaussiano; verificar se Monte Carlo ainda acompanha o DOP.
- [ ] (Opcional) Mostrar uma extensão: restrição de raio máximo (Gueye et al.) ou mínimos quadrados robustos.
- [ ] Gerar gráficos e mapas finais (alvo real, estimado, círculos de raio `v·t` dos servidores).

### Fase 10 — Relatório final (rascunho incremental desde 16/11; versão final até 01/12)
Estrutura sugerida (ver seção 6).

### Fase 11 — Revisão e apresentação (01/12–04/12)
- [ ] 01/12–02/12: revisão cruzada do texto e do código; reexecutar todos os experimentos a partir do zero.
- [ ] 03/12: ensaio da apresentação com cronometragem.
- [ ] Slides (~12–15): problema, modelo, método, geometria/DOP, QR vs. normais, Monte Carlo, dados reais, limitações, conclusão.
- [ ] Ensaiar apresentação; entregar.

---

## 5. Plano de testes

| Teste | O que verifica | Critério |
|---|---|---|
| `test_geo` | haversine vs. distâncias conhecidas | erro < 1 km |
| `test_jacobiana` | derivada analítica vs. diferenças finitas | erro rel. < 1e-6 |
| `test_solvers` | Cholesky/QR vs. `numpy.linalg.lstsq` | erro rel. < 1e-10 (float64, κ moderado) |
| `test_gauss_newton` | dados sem ruído | recupera θ com erro < 1e-6 |
| `test_ortogonalidade` | `QᵀQ ≈ I`, `QR ≈ G` | erro < 1e-12 |
| `test_dop` | caso simples analítico (ex.: 4 servidores simétricos) | valor esperado |

---

## 6. Esboço do relatório final

1. **Introdução** — problema, motivação, analogia com GPS.
2. **Modelagem** — RTT→distância, atraso `b`, esfera, mínimos quadrados não linear.
3. **Métodos numéricos** — Gauss-Newton, equações normais, QR/Householder, critério de parada.
4. **Geometria e condicionamento** — DOP, κ₂(G), relação com erro.
5. **Experimentos sintéticos** — convergência, normais vs. QR (float32/64), Monte Carlo.
6. **Experimentos com dados reais** — coleta, pré-processamento, resultados.
7. **Discussão e limitações** — inflação de latência, rotas, assimetria, anycast.
8. **Conclusão** — resposta à pergunta de pesquisa, trabalhos futuros.
9. **Referências** (ABNT, as 7 do projeto) e **Apêndice** (instruções para reproduzir).

### Figuras/tabelas obrigatórias
- Mapa com servidores, alvo real e estimativa.
- ‖δₖ‖ × iteração para cada chute inicial.
- Erro relativo × κ (normais vs. QR, float32 vs. float64).
- Tabela geometria × (PDOP, GDOP, κ₂, erro).
- Erro Monte Carlo × erro previsto (PDOP·σ).
- Resíduos reais por servidor (evidenciando viés positivo).

---

## 7. Riscos e mitigação

| Risco | Impacto | Mitigação |
|---|---|---|
| Gauss-Newton diverge ou cai em mínimo local com chute ruim | Alto | Reportar como resultado; opcionalmente adicionar amortecimento (LM) ou chute inicial por centroide ponderado |
| Jacobiana errada | Alto | Teste obrigatório com diferenças finitas |
| Divergência float32 não aparece nos κ testados | Médio | Estender a faixa de κ; usar matrizes sintéticas via SVD |
| Anycast/CDN distorce RTT | Médio | Preferir universidades e sondas RIPE com localização verificada; checar RTT mínimo plausível |
| Poucos servidores com boa geometria | Médio | Incluir servidores em todos os continentes; usar RIPE Atlas |
| Erro real muito maior que o previsto | Esperado | É resultado válido: explicar pela inflação e assimetria (objetivo 7) |
| Atraso no cronograma (prazo apertado: ~5,5 semanas) | Alto | Congelar código em 26/11; paralelizar Fases 6/7/8/10; cortar extensões opcionais (LM, CBG, Gram-Schmidt) se atrasar mais de 2 dias nos marcos de 06/11 ou 13/11 |
| Poucos dados reais por coleta tardia | Médio | Iniciar o script de ping em 28/10 e deixá-lo rodando em segundo plano |
| Divergência de resultados entre membros | Baixo | Sementes fixas, scripts que geram tudo, Git |

---

## 8. Checklist final de entrega
- [ ] Código no repositório, rodando com `pip install -r requirements.txt` + um comando por experimento.
- [ ] Todos os testes passando (`pytest`).
- [ ] `results/` com todas as figuras do relatório.
- [ ] Relatório em PDF com as referências.
- [ ] Slides da apresentação.
- [ ] Todos os integrantes com contribuição registrada no Git.
