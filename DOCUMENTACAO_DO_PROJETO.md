# Documentação do Projeto: Geolocalização de IP por Latência de Rede
**Trabalho Final de Cálculo Numérico**  
*Autores: Gabriel Santana, Luís Fernando, Lucas dos Santos, Luís Gustavo*  
*Data de Atualização: 29 de Outubro de 2026*  
*Status do Projeto: Fases 1, 2, 3 e 7 concluídas com 100% dos testes aprovados.*

---

## 1. Visão Geral e Alinhamento com Cálculo Numérico

O objetivo do projeto é estimar a localização geográfica $(\phi, \lambda)$ e o atraso fixo de processamento de rede $b$ a partir de medições de tempo de ida e volta (RTT - *Round Trip Time*) de pings para servidores com coordenadas conhecidas.

O problema é formulado estritamente como um **Problema de Mínimos Quadrados Não-Linear**:
$$\min_{\theta} \frac{1}{2} \|r(\theta)\|_2^2, \quad \text{onde } \theta = (\phi, \lambda, \beta)^T \text{ e } \beta = v \cdot b \text{ (em km)}$$

### Por que o projeto atende com rigor a ementa da disciplina:
1. **Gauss-Newton:** Extensão direta do método de Newton-Raphson multivariado para sistemas sobredeterminados ($m > n$), linearizando resíduos e resolvendo mínimos quadrados lineares a cada iteração.
2. **Equações Normais e Fatoração de Cholesky:** Algoritmo clássico de Banachiewicz para resolução de sistemas simétricos definidos positivos $(G^T G) \delta = -G^T r$.
3. **Fatoração QR via Reflexões de Householder:** Solução numericamente estável para mínimos quadrados $R \delta = -Q^T r$ que preserva o número de condição ($\kappa(G)$ em vez de $\kappa(G)^2$).
4. **Precisão de Ponto Flutuante (`float32` vs `float64`):** Análise experimental do impacto da perda de dígitos significativos por cancelamento e condicionamento.
5. **Diferenciação Numérica:** Validação da Jacobiana analítica por Diferenças Finitas Centrais.

---

## 2. Estrutura de Arquivos Criada no Projeto

A pasta do projeto foi estruturada na Área de Trabalho em `C:\Users\User\Desktop\Projeto_Calculo_Numerico_GeoIP`:

```text
Projeto_Calculo_Numerico_GeoIP/
├── servers.csv                     # Lista de servidores verificados com lat/lon
├── coletor_pings.py                # Script em Python de ping periódico
├── iniciar_coleta_continua.bat     # Coletor automático a cada 10 minutos
├── executar_coleta_uma_vez.bat     # Execução pontual de teste
├── SDD_Projeto_Geolocalizacao_IP.md # Documento de Desenho de Software (SDD)
├── painel_projeto_geoip.html       # Painel interativo com checklist e decisões
├── DOCUMENTACAO_DO_PROJETO.md      # Este relatório de progresso
│
├── src/                            # Módulos matemáticos do projeto
│   ├── __init__.py
│   ├── geo.py                      # Haversine, conversões e vetores unitários
│   ├── model.py                    # Função de resíduos r(θ) e Jacobiana G(θ)
│   ├── solvers.py                  # Cholesky, Substituições e Householder QR
│   └── gauss_newton.py             # Laço principal iterativo de Gauss-Newton
│
├── tests/                          # Suíte de testes unitários automatizados (pytest)
│   ├── __init__.py
│   ├── test_geo.py                 # 7 testes de trigonometria esférica
│   ├── test_jacobiana.py           # 3 testes de derivada analítica vs finita
│   ├── test_solvers.py             # 7 testes de álgebra linear e decomposições
│   └── test_gauss_newton.py        # 4 testes de convergência com dados sintéticos
│
└── logs_pings/                     # Armazenamento contínuo das medições de rede
    ├── historico_pings.csv         # Tabela consolidada de medições
    └── ping_raw_YYYYMMDD_HHMMSS.txt# Saída textual bruta dos comandos de ping
```

---

## 3. Detalhamento dos Módulos Implementados

### 3.1. Módulo Geográfico (`src/geo.py`)
* **`haversine(lat1, lon1, lat2, lon2, R=6371.0)`**:
  Calcula a distância ortodrômica de círculo máximo em km. Utiliza formulação numéricamente estável com $\text{atan2}$ para evitar indefinições numéricas do $\arccos$ em distâncias pequenas. Suporta arrays NumPy vetorizados.
* **`latlon_to_unit_vector(lat, lon)`**:
  Mapeia $(\phi, \lambda)$ para coordenadas cartesianas na esfera unitária tridimensional:
  $$u = [\cos\phi \cos\lambda, \cos\phi \sin\lambda, \sin\phi]^T$$
* **`azimute_inicial(lat1, lon1, lat2, lon2)`**:
  Ângulo de visada entre o alvo e o servidor para determinação da geometria.

### 3.2. Módulo de Modelagem (`src/model.py`)
* **`residuals(theta, servers, rtt_ms, v=100.0)`**:
  Vetor de resíduos de dimensão $(m,)$ em km:
  $$r_i(\theta) = d_i(\phi, \lambda) + \beta - v \cdot t_i$$
  onde $v = 100\text{ km/ms}$ representa aproximadamente dois terços da velocidade da luz na fibra óptica.
* **`jacobian_analytical(theta, servers, v=100.0)`**:
  Dedução formal da matriz Jacobiana $G \in \mathbb{R}^{m \times 3}$:
  $$G[i, :] = \left[ \frac{\partial d_i}{\partial \phi}, \frac{\partial d_i}{\partial \lambda}, 1.0 \right]$$
  Calculada pelas projeções dos vetores unitários na base do plano tangente local (Norte e Leste).
* **`jacobian_finite_diff(theta, servers, v=100.0, h_deg=1e-6)`**:
  Cálculo das derivadas parciais via Diferenças Finitas Centrais:
  $$\frac{\partial r}{\partial \phi} \approx \frac{r(\phi + h, \lambda, \beta) - r(\phi - h, \lambda, \beta)}{2h}$$

### 3.3. Módulo de Solvers Numéricos (`src/solvers.py`)
* **Substituição Progressiva (`forward_substitution`)**: Resolve $L y = b$ para matriz triangular inferior em $\mathcal{O}(n^2)$.
* **Substituição Regressiva (`back_substitution`)**: Resolve $U x = b$ para matriz triangular superior em $\mathcal{O}(n^2)$.
* **Fatoração de Cholesky (`cholesky`)**:
  Decompõe $A = L L^T$ pelo algoritmo de Banachiewicz. Detecta se a matriz não é positiva definida.
* **Solução via Equações Normais (`solve_normal`)**:
  Monta $A = G^T G$ e $b = -G^T r$, resolvendo por Cholesky. Suporta `float32` e `float64`.
* **Fatoração QR de Householder (`householder_qr`)**:
  Gera matrizes ortogonais $H_k = I - 2 v_k v_k^T$ com controle de cancelamento catastrófico no cálculo do escalar $\alpha$.
* **Solução via QR (`solve_qr`)**:
  Projeta $d = -Q^T r$ e resolve o sistema triangular superior $R \delta = d$ por substituição regressiva.

### 3.4. Laço Iterativo de Gauss-Newton (`src/gauss_newton.py`)
* Executa as iterações de Gauss-Newton:
  $$\theta_{k+1} = \theta_k + \delta_k$$
* Inclui amortecimento angular / projeção esférica (`max_step_deg = 15.0°`) para evitar instabilidade trigonométrica em grandes distâncias.
* **Critérios de Parada Rigorosos:**
  1. Passo pequeno: $\|\delta_k\| < \text{tol}_x$ ($10^{-3}\text{ km} = 1\text{ metro}$).
  2. Estagnação de resíduo: $|\|r_k\| - \|r_{k-1}\|| / \|r_k\| < \text{tol}_f$ ($10^{-8}$).
  3. Limite de iterações: $k \ge 50$.
  4. Detecção de divergência: $\|r_k\|$ crescendo por 3 iterações consecutivas.
* Retorna o objeto `GaussNewtonResult` com histórico completo de cada iteração ($\theta_k, \|r_k\|, \|\delta_k\|, \kappa_2(G)$).

---

## 4. Resultados da Suíte de Testes Automatizados (`pytest`)

Execução de todos os 21 testes unitários com gabarito de bibliotecas padrão:

```text
tests/test_gauss_newton.py::test_gauss_newton_recuperacao_exata_qr PASSED
tests/test_gauss_newton.py::test_gauss_newton_recuperacao_exata_normais PASSED
tests/test_gauss_newton.py::test_convergencia_multiplos_chutes PASSED
tests/test_gauss_newton.py::test_historico_iteracoes PASSED
tests/test_geo.py::test_distancia_identidade PASSED
tests/test_geo.py::test_distancia_simetria PASSED
tests/test_geo.py::test_distancia_um_grau_equador PASSED
tests/test_geo.py::test_distancia_conhecida_sp_rio PASSED
tests/test_geo.py::test_distancia_conhecida_sp_lisboa PASSED
tests/test_geo.py::test_haversine_vetorizado PASSED
tests/test_geo.py::test_vetor_unitario PASSED
tests/test_jacobiana.py::test_dimensoes_residuos_e_jacobiana PASSED
tests/test_jacobiana.py::test_jacobiana_analitica_vs_diferencas_finitas PASSED (erro < 1e-9)
tests/test_jacobiana.py::test_consistencia_residuos PASSED
tests/test_solvers.py::test_cholesky_decomposicao PASSED
tests/test_solvers.py::test_substituicao_progressiva_e_regressiva PASSED
tests/test_solvers.py::test_ortogonalidade_e_reconstrucao_qr PASSED (erro < 1e-12)
tests/test_solvers.py::test_solve_normal_vs_numpy_lstsq PASSED (erro < 1e-10)
tests/test_solvers.py::test_solve_qr_vs_numpy_lstsq PASSED (erro < 1e-10)
tests/test_solvers.py::test_concordancia_normais_vs_qr_em_float64 PASSED
tests/test_solvers.py::test_suporte_float32 PASSED

============================= 21 passed in 0.37s =============================
```

---

## 5. Fase 7: Coleta de Dados Reais de Ping

* **Local de Coleta Fixo Definido:** Conexão do ambiente de trabalho do integrante em São Paulo (6 horas contínuas de coleta diária).
* **Frequência da Coleta:** A cada 10 minutos (600 segundos), via `iniciar_coleta_continua.bat`.
* **Servidores Ativos e Validados (100% de resposta ICMP IPv4):**
  1. USP (IME - São Paulo, SP): ~5.0 ms
  2. Unicamp (IC - Campinas, SP): ~7.0 ms
  3. UFRJ (Rio de Janeiro, RJ): ~11.0 ms
  4. UFMG (DCC - Belo Horizonte, MG): ~16.0 ms
  5. NTP.br (Curitiba, PR): ~18.0 ms
  6. UFRGS (INF - Porto Alegre, RS): ~20.0 ms
  7. UnB (Brasília, DF): ~20.0 ms
  8. NTP.br (Fortaleza, CE): ~52.0 ms
  9. CMU (Pittsburgh, EUA): ~121.0 ms
  10. NIC.br / Registro.br (São Paulo, SP): ~4.0 ms
  11. NTP.br (São Paulo, SP): ~4.0 ms

---

## 6. Próximos Passos Imediatos

Com os algoritmos de Gauss-Newton, Cholesky e QR prontos e testados, as próximas tarefas imediatas são:
1. **Fase 3 / Exp01:** Gerar os gráficos de convergência de Gauss-Newton ($\log \|\delta_k\|$ vs $k$) para múltiplos chutes iniciais.
2. **Fase 4 / Exp02:** Construir o experimento comparativo entre Equações Normais e QR em matrizes com condicionamento controlado $\kappa \in [10^0, 10^{10}]$ em `float32` e `float64` (gráfico de 4 curvas em log-log).
3. **Fase 5 / Exp03:** Implementar a métrica de Diluição de Precisão (DOP) em `src/dop.py`.
