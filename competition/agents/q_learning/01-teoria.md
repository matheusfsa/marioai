# Q-Learning — Teoria

**Q-Learning** é um algoritmo de RL tabular **TD(0) off-policy**, proposto por Watkins em 1989. É o "arroz-e-feijão" do RL clássico: simples, robusto, com prova de convergência para `Q*` sob condições brandas.

## TD(0) off-policy

A equação de update é:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a′} Q(s′, a′) − Q(s, a)]
```

Compare com SARSA: onde SARSA usa `Q(s′, a′)` com `a′` amostrado pela política, Q-learning usa `max_{a′} Q(s′, a′)` — ou seja, assume que no próximo passo a ação ótima **será** tomada, independentemente da política de exploração atual.

Este é o significado de **off-policy**: aprende a política gulosa ótima (`Q*`) enquanto segue uma política diferente (tipicamente ε-greedy).

## Por que funciona

O operador `max_{a′} Q(s′, a′)` aplicado à equação de Bellman ótima:

```
Q*(s, a) = E[r + γ · max_{a′} Q*(s′, a′)]
```

é uma contração no sup-norm, então a iteração converge para `Q*`. Em prática, com α apropriado (somável mas não quadraticamente) e toda (s, a) visitada infinitamente, `Q → Q*` com probabilidade 1 (Watkins & Dayan, 1992).

## Pseudocódigo

```
inicializa Q[s, a] = 0 para todo s, a
para episódio = 1..E:
    s ← estado inicial
    enquanto episódio não termina:
        a ← ε-greedy(Q, s)
        executa a, observa r, s′
        Q[s, a] += α · (r + γ · max_{a′} Q[s′, a′] − Q[s, a])
        s ← s′
    decai ε
```

## Exploração vs. exploração

- **ε-greedy** é o padrão: com probabilidade `ε` explora, caso contrário explota `argmax`.
- **Boltzmann / Softmax**: amostra `a` com probabilidade `exp(Q(s, a) / τ) / Z` — temperatura `τ` controla aleatoriedade.
- **UCB**: seleciona `a = argmax [Q(s, a) + c · √(ln N(s) / N(s, a))]` — bônus para ações pouco exploradas.

Para Mario, ε-greedy com decay linear é o suficiente.

## Vantagens

- **Off-policy** → separação clara entre exploração e explotação aprendida.
- Aprende a política ótima (sob condições).
- Atualização por passo → mais eficiente amostralmente que MC.

## Desvantagens

- **Maximization bias**: o operador `max` tende a superestimar valores quando Q tem ruído (resolvido por Double Q-learning, Hasselt 2010).
- Mesmo problema de escalabilidade de toda Q-table: número de estados explode com features contínuas ou muitas dimensões.

## Double Q-Learning (menção)

Para reduzir o bias do max, mantém duas tabelas `Q_A` e `Q_B`, e alterna:

```
Q_A(s, a) ← Q_A(s, a) + α · [r + γ · Q_B(s′, argmax_{a′} Q_A(s′, a′)) − Q_A(s, a)]
```

Não usado no agente da competição, mas é a base do **Double DQN**.

## Equivalência com Bellman

A equação de Bellman ótima para Q é:

```
Q*(s, a) = R(s, a) + γ · Σ_{s′} P(s′ | s, a) · max_{a′} Q*(s′, a′)
```

Q-learning é essencialmente iteração de valor estocástica aplicada a Q — cada update é uma amostra do operador de Bellman.

## Referências

- Watkins, C. J. C. H. (1989). *Learning from delayed rewards*. PhD thesis, Cambridge.
- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3-4), 279–292.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, cap. 6.5.
- Hasselt, H. V. (2010). *Double Q-learning*. NIPS.
