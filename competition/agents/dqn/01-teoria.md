# DQN (Deep Q-Network) — Teoria

**DQN** (Mnih et al., 2015) é a extensão de Q-learning para espaços de estado contínuos ou de alta dimensionalidade. Substitui a Q-table por uma **rede neural** `Q(s, a; θ)` e introduz dois truques essenciais para estabilizar o treinamento: **replay buffer** e **target network**.

## Por que redes neurais

Q-learning tabular exige enumerar estados — em Mario com pixels ou grids grandes, o espaço é astronômico. Uma rede neural aproxima `Q(s, a)` como função diferenciável dos pixels/tiles de entrada e generaliza entre estados similares sem visitá-los todos.

## Arquitetura típica

Para entradas tipo "imagem" (frames 22×22 ou 84×84):

```
entrada: (N, C_hist, H, W)        // C_hist = frames empilhados
  → Conv2D(32, kernel=3, stride=1) + ReLU
  → Conv2D(64, kernel=3, stride=1) + ReLU
  → Flatten
  → Linear(256) + ReLU
  → Linear(|A|)                    // uma Q-value por ação
```

Frame stacking (tipicamente 4) resolve parcialmente o problema de POMDP: mostrar 4 frames consecutivos dá ao agente informação de velocidade implícita.

## Loss

Para cada transição `(s, a, r, s′, d)` amostrada do replay buffer:

```
y = r + γ · (1 − d) · max_{a′} Q(s′, a′; θ⁻)
L(θ) = E[(y − Q(s, a; θ))²]
```

Onde:
- `θ` é a rede online (atualizada a cada gradient step).
- `θ⁻` é a **target network**, cópia atrasada de `θ` (sincroniza a cada N passos).
- `d ∈ {0, 1}` é o flag de terminal (`1` se episódio acabou em `s′`).

## Replay Buffer

Guarda transições `(s, a, r, s′, d)` numa fila circular de capacidade fixa (ex.: 100k a 1M). A cada gradient step, amostra um mini-batch (tipicamente 32) uniformemente. Benefícios:

- **Quebra correlação temporal**: transições consecutivas são altamente correlacionadas; amostragem aleatória aproxima a premissa i.i.d. do SGD.
- **Reuso de experiência**: cada transição é usada múltiplas vezes, melhorando eficiência amostral.

## Target Network

A segunda rede `θ⁻` é uma cópia de `θ` sincronizada a cada N passos (ex.: 1000). Ao usar `θ⁻` no target, congelamos o alvo de regressão por um intervalo, quebrando o feedback loop `θ` estimando `θ` — que tende a divergir.

## Pseudocódigo

```
inicializa θ (rede online), θ⁻ ← θ (target)
inicializa replay D (capacidade N_rep)
para passo = 1..T:
    s = observa()
    a = ε-greedy(Q(s, ·; θ))
    executa a, observa r, s′, d
    D.append((s, a, r, s′, d))
    s ← s′ (ou reset se d)

    se |D| ≥ batch e passo % update_freq == 0:
        amostra B = {(s, a, r, s′, d)} ~ D
        y = r + γ · (1 − d) · max_{a′} Q(s′, a′; θ⁻)
        loss = mean((y − Q(s, a; θ))²)
        θ ← θ − α · ∇θ loss

    se passo % target_sync == 0:
        θ⁻ ← θ
    decai ε
```

## Melhorias clássicas (variantes)

| Variante | Artigo | Ideia |
|---|---|---|
| Double DQN | Hasselt et al. 2015 | Usa `θ` para seleção, `θ⁻` para avaliação — reduz maximization bias |
| Dueling DQN | Wang et al. 2015 | Separa rede em `V(s)` e `A(s, a)` antes de somar |
| Prioritized Replay | Schaul et al. 2015 | Amostra transições com TD-error alto mais frequentemente |
| Noisy Nets | Fortunato et al. 2017 | Exploração via ruído parametrizado na rede |
| Rainbow | Hessel et al. 2017 | Combina todas as melhorias acima |

## Hiperparâmetros sensíveis

- **Learning rate** (1e-4 a 5e-4 comum).
- **γ** (0.99 padrão).
- **Batch size** (32 a 256).
- **Target sync frequency** (100 a 10000 passos).
- **Replay capacity** (tem que ser maior que o quadrado da correlação temporal típica).
- **Exploration schedule** (decay linear é o suficiente na maioria dos casos).

## Trade-offs vs. tabular

| Vantagem | Desvantagem |
|---|---|
| Escala para espaços contínuos/altos | Instável; exige replay + target |
| Generaliza entre estados | Precisa GPU para ser prático |
| Sem feature engineering | Muitos hiperparâmetros |
| Transferível com fine-tuning | Nenhuma garantia de convergência |

## Referências

- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533.
- Hasselt, H. V., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with double Q-learning*. AAAI.
- Hessel, M., et al. (2018). *Rainbow: Combining improvements in deep reinforcement learning*. AAAI.
