# Monte Carlo Control — Teoria

**Monte Carlo (MC) control** é o algoritmo clássico de RL tabular que aprende uma política ótima usando **retornos de episódios completos**, sem bootstrap. "Monte Carlo" vem de simular trajetórias e calcular médias empíricas.

## Por que Monte Carlo

Em TD(0) (SARSA / Q-learning), o agente atualiza Q após **cada passo** usando a estimativa do próximo estado — introduz viés mas reduz variância. MC vai no extremo oposto: **espera o episódio terminar** e usa o retorno real observado. Alto variância, zero viés. Convergência mais lenta em média, mas sem erros acumulados de bootstrap.

## Retorno descontado

O retorno a partir do passo `t` é a soma descontada dos rewards futuros:

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... = Σ_{k=0}^{T−t−1} γ^k · r_{t+k+1}
```

`γ ∈ [0, 1]` é o fator de desconto: `γ=1` trata o episódio todo uniformemente; `γ<1` dá mais peso ao reward imediato.

## Algoritmo (on-policy, ε-greedy, first-visit)

```
inicializa Q[s, a] = 0, N[s, a] = 0 para todo s, a
π ← ε-greedy(Q)
para episódio = 1..E:
    gera trajetória (s₀, a₀, r₁, s₁, a₁, r₂, …, s_T) seguindo π
    G = 0
    para t = T−1, T−2, ..., 0:
        G = γ·G + r_{t+1}
        se (s_t, a_t) aparece primeiro em t (first-visit):
            N[s_t, a_t] += 1
            Q[s_t, a_t] += (G − Q[s_t, a_t]) / N[s_t, a_t]
    atualiza π ← ε-greedy(Q)
    decai ε
```

**First-visit** vs. **every-visit**: na first-visit, só se atualiza o par `(s, a)` na primeira vez que aparece no episódio; na every-visit, em toda ocorrência. Convergência comprovada em ambos; first-visit tem menos viés amostral.

## Policy improvement (GLIE)

Sob GLIE (Greedy in the Limit with Infinite Exploration — `ε → 0` mas toda ação sempre tem probabilidade > 0), MC-ε-greedy converge para a política ótima `π*` e seu Q ótimo `Q*`.

## Vantagens

- Sem bootstrap → **sem viés** de estimativa.
- Converge em ambientes **não-markovianos** (desde que a política dependa só do estado).
- Não precisa de modelo do ambiente.
- Fácil de implementar: loop de episódio + média incremental.

## Desvantagens

- **Só aprende no fim do episódio** — ruim para tarefas longas (em Mario, um episódio leva segundos de jogo).
- **Alta variância** — o retorno depende de toda a trajetória futura.
- Não aprende em tarefas contínuas (sem término natural).

## Equação de update incremental

A média amostral incremental é:

```
Q(s, a) ← Q(s, a) + (1 / N(s, a)) · [G − Q(s, a)]
```

Com `α` constante (em vez de `1/N`), vira média exponencialmente ponderada — útil para não-estacionariedade.

## Pseudocódigo de avaliação

```
ε ← 0
para cada fase:
    reset ambiente
    enquanto episódio não termina:
        s = observa()
        a = argmax_a Q[s, a]
        passo(a)
```

## Referências

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, cap. 5 — Monte Carlo methods.
- Singh, S., et al. (2000). *Convergence results for single-step on-policy reinforcement-learning algorithms*. Machine Learning, 38(3).
