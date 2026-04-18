# SARSA — Teoria

**SARSA** (State-Action-Reward-State-Action) é um algoritmo de RL tabular **TD(0) on-policy**. O nome vem da tupla que define uma atualização: `(s, a, r, s′, a′)`. Aprende Q da política que está seguindo — diferente do Q-learning, que aprende Q da política ótima independente do que está seguindo.

## TD(0) on-policy

A ideia central do aprendizado por diferenças temporais: em vez de esperar o fim do episódio (MC), atualiza `Q(s, a)` imediatamente usando a estimativa do próximo par `(s′, a′)` **amostrado pela política atual**.

## Equação de update

```
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s′, a′) − Q(s, a)]
```

Onde:
- `α ∈ (0, 1]` é a taxa de aprendizado.
- `γ ∈ [0, 1]` é o fator de desconto.
- `(s′, a′)` é o próximo par estado-ação **realmente tomado** pela política.
- O termo `r + γ · Q(s′, a′) − Q(s, a)` é o **erro TD**.

## Por que "on-policy"

Como `a′` é amostrado pela política atual (ε-greedy, por exemplo), SARSA aprende o valor da política ε-greedy, não da política ótima. Isto tem uma consequência prática importante: **SARSA considera o ruído da exploração**.

Exemplo clássico (Sutton & Barto, cliff walking): numa penhasco com reward muito negativo à beira, SARSA aprende um caminho **seguro** (longe da beira), porque o "ruído" do ε pode empurrar o agente para o precipício. Q-learning, off-policy, aprende o caminho **ótimo** (rente à beira) porque ignora o ruído — mas na prática, seguindo uma política ε-greedy, ele cai no penhasco.

## Pseudocódigo

```
inicializa Q[s, a] = 0 para todo s, a
para episódio = 1..E:
    s ← estado inicial
    a ← ε-greedy(Q, s)
    enquanto episódio não termina:
        executa a, observa r, s′
        a′ ← ε-greedy(Q, s′)
        Q[s, a] += α · (r + γ · Q[s′, a′] − Q[s, a])
        s, a ← s′, a′
    decai ε
```

## Convergência

Sob condições padrão (α somável mas não quadraticamente, toda (s, a) visitada infinitamente, GLIE), SARSA converge para `Q*` da política ótima.

## Variantes

- **Expected SARSA**: em vez de `Q(s′, a′)`, usa `E_π [Q(s′, a′)] = Σ_a π(a|s′) Q(s′, a)`. Reduz variância.
- **SARSA(λ)**: generaliza para TD(λ) com traces de elegibilidade — interpola entre TD(0) e MC.

## Diferença vs. Q-learning

| | SARSA | Q-learning |
|---|---|---|
| Tipo | On-policy | Off-policy |
| Target | `Q(s′, a′)` com `a′ ~ π` | `max_a' Q(s′, a′)` |
| Aprende | Q da política atual | Q da política ótima |
| Comportamento | Mais seguro com exploração | Mais agressivo |

## Trade-offs

| Vantagem | Desvantagem |
|---|---|
| Atualiza a cada passo (mais eficiente amostralmente) | Bootstrap introduz viés |
| Mais "conservador" sob ε > 0 | Depende da política de exploração |
| Simples e estável na prática | Convergência para `Q*` só com ε → 0 |

## Referências

- Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems*. Tech. Report CUED/F-INFENG/TR 166 — artigo que introduziu o nome "modified connectionist Q-learning", mais tarde apelidado SARSA.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, cap. 6 — Temporal-Difference Learning.
