# PPO (Proximal Policy Optimization) — Teoria

**PPO** (Schulman et al., 2017) é um algoritmo de RL **policy-gradient** on-policy, derivado do TRPO (Trust Region Policy Optimization). Virou referência em deep RL por ser estável, simples de implementar e funcionar bem sem tuning agressivo. Usado em benchmarks desde Atari até controle robótico e alinhamento de LLMs (RLHF).

## Policy Gradient vs. Valor

Enquanto DQN aprende `Q(s, a)` e deriva a política como `argmax`, PPO aprende **diretamente a política** `π(a | s; θ)` como distribuição parametrizada. Vantagens:

- Espaços de ação contínuos são naturais.
- Políticas estocásticas (úteis em jogos multi-agente ou com observação parcial).
- Gradiente direto do objetivo.

## Objetivo policy-gradient básico

```
J(θ) = E_{τ ~ π_θ} [Σ_t r_t]
∇J(θ) = E[Σ_t ∇log π_θ(a_t | s_t) · A_t]
```

`A_t` é a **vantagem** — quanto melhor (ou pior) `a_t` foi comparado à média em `s_t`. Estimada via baseline (função de valor) para reduzir variância.

## GAE (Generalized Advantage Estimation)

Schulman et al. 2015 — interpola entre TD(0) e MC:

```
δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
A_t^GAE(λ) = Σ_{k=0}^∞ (γλ)^k · δ_{t+k}
```

`λ = 0` → TD(0) (viés alto, variância baixa); `λ = 1` → MC (viés zero, variância alta). Valor típico: `λ = 0.95`.

## O truque do PPO: *clipping*

Policy gradient ingênuo é instável porque atualizações grandes podem colapsar a política. TRPO resolve com restrição de KL (cara). PPO simplifica: clipa a razão de probabilidades.

Seja `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`. A loss clipada é:

```
L^CLIP(θ) = E_t [ min(r_t(θ) · A_t, clip(r_t(θ), 1−ε, 1+ε) · A_t) ]
```

Com `ε` tipicamente 0.1 ou 0.2. Intuição:
- Se `A_t > 0` e `r_t` está crescendo além de `1+ε`, o clip impede ganho extra — não encoraja passos muito grandes.
- Se `A_t < 0` e `r_t` está caindo abaixo de `1−ε`, idem no outro lado.

## Loss total

PPO otimiza três termos conjuntamente:

```
L(θ, φ) = L^CLIP(θ) − c_1 · (V_φ(s) − V_target)² + c_2 · H[π_θ(·|s)]
```

- `L^CLIP`: objetivo de política (maximizar).
- `(V − V_target)²`: loss da função de valor (minimizar).
- `H[π]`: entropia da política (maximizar) — incentiva exploração.

`c_1 ≈ 0.5`, `c_2 ≈ 0.01` são padrão.

## Pseudocódigo

```
inicializa π_θ, V_φ
para iteração = 1..N:
    coleta T passos com π_θ_old → trajetórias {(s, a, r, s′)}
    calcula A_t via GAE e V_target = A_t + V_φ(s_t)
    para época = 1..K:
        para minibatch em trajetórias:
            r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            L_CLIP = min(r_t · A_t, clip(r_t, 1−ε, 1+ε) · A_t)
            L_V    = (V_φ(s_t) − V_target)²
            L_H    = −H[π_θ(·|s_t)]
            θ, φ   ← SGD sobre −(L_CLIP − c_1·L_V − c_2·L_H)
    π_θ_old ← π_θ
```

## Hiperparâmetros principais

| Nome | Típico | Papel |
|---|---|---|
| `n_steps` | 2048 | Passos coletados por iteração (rollout) |
| `n_epochs` | 4–10 | Epochs de SGD sobre o rollout coletado |
| `batch_size` | 64 | Minibatch dentro do rollout |
| `γ` | 0.99 | Desconto |
| `λ` (GAE) | 0.95 | Tradeoff viés/variância |
| `clip_range` | 0.2 | Limite da razão `r_t` |
| `ent_coef` | 0.01 | Peso da entropia |
| `vf_coef` | 0.5 | Peso da loss de valor |
| `lr` | 3e-4 | Learning rate Adam |

## PPO vs. DQN

| | PPO | DQN |
|---|---|---|
| Tipo | Policy-based | Value-based |
| On/off-policy | On-policy | Off-policy |
| Ações | Contínuas ou discretas | Discretas |
| Replay buffer | Não (rollout descartado) | Sim |
| Estabilidade | Geralmente boa | Sensível a hiperparâmetros |
| Eficiência amostral | Pior (não reusa experiência) | Melhor |

## Quando preferir PPO

- Ações contínuas (DQN não funciona diretamente).
- Política estocástica desejada.
- Ambiente com "paralelização barata" (múltiplos envs coletando rollouts). **Em Mario isso é um problema** — o servidor Java aceita só uma conexão por vez na porta 4242.

## Referências

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Schulman, J., et al. (2015). *High-dimensional continuous control using generalized advantage estimation*. ICLR 2016.
- Achiam, J. (2018). *Spinning Up in Deep RL* — [https://spinningup.openai.com](https://spinningup.openai.com).
