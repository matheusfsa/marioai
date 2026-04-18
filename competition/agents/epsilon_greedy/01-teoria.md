# ε-greedy com features — Teoria

Estratégia de **exploração vs. exploração** mais simples em RL: com probabilidade `ε`, escolhe uma ação aleatória; com probabilidade `1 − ε`, escolhe a ação com maior valor estimado (`argmax_a Q(s, a)`). Combinada com uma Q-table indexada por features discretas, forma um agente de aprendizado simples mas funcional.

## Intuição

Todo agente de RL enfrenta o dilema: explorar ações pouco testadas (podem ser melhores) ou explotar o que já parece bom? A escolha uniformemente aleatória dominada por `ε` é subótima em teoria, mas boa na prática — especialmente quando `ε` decresce com o tempo.

## Definições

- `Q(s, a)` — valor esperado do retorno ao tomar ação `a` no estado `s`.
- Política ε-greedy: `π(a | s) = 1 − ε + ε/|A|` se `a = argmax_a Q(s, a)`, caso contrário `ε/|A|`.
- `ε-schedule`: função que reduz `ε` ao longo do treino (linear, exponencial, ou por desempenho).

## Atualização (variante Monte Carlo sem desconto)

Nesta variante, usamos retornos empíricos ao fim do episódio (sem bootstrap TD):

```
Para cada par (s, a) visitado no episódio:
    G = soma dos rewards futuros a partir daquele passo
    N(s, a) ← N(s, a) + 1
    Q(s, a) ← Q(s, a) + (1 / N(s, a)) · (G − Q(s, a))
```

Isto é idêntico a uma média amostral incremental: `Q` converge para `E[G | s, a]` sob a política atual.

## Pseudocódigo

```
inicializa Q[s, a] = 0 para todo s, a
inicializa N[s, a] = 0
ε ← 1.0

para episódio = 1..E:
    reset ambiente
    trajetória = []
    enquanto episódio não termina:
        s = observa()
        se rand() < ε:
            a = aleatório(|A|)
        senão:
            a = argmax_a Q[s, a]
        r, s' = passo(a)
        trajetória.append((s, a, r))
    # atualização ao fim do episódio
    G = 0
    para (s, a, r) em trajetória (invertido):
        G = G + r
        N[s, a] += 1
        Q[s, a] += (G − Q[s, a]) / N[s, a]
    ε ← max(ε_min, ε − Δε)
```

## Por que funciona

Dado que toda ação é tentada em todo estado infinitamente (na média, com ε > 0), `N(s, a) → ∞` e `Q(s, a) → E[G | s, a]`. Isto é *GLIE* (Greedy in the Limit with Infinite Exploration) — teorema clássico que garante convergência para a política ótima.

## Discretização de features (requisito)

Q-table requer **estado hashable**. Em Mario, usamos features booleanas (ex.: "inimigo à frente", "buraco à frente") combinadas numa tupla. Cada combinação é um "estado" discreto. Quanto mais features, maior a tabela e mais lento converge — há um trade-off entre granularidade e velocidade de aprendizado.

## Diferença vs. Monte Carlo puro

- Retornos **não descontados** (γ = 1).
- Sem condição de "first-visit" — atualiza a cada visita ao par (s, a).

São variantes menores; a ideia central é a mesma: learner por trajetória completa com política ε-greedy.

## Trade-offs

| Vantagem | Desvantagem |
|---|---|
| Muito simples de implementar | Requer features projetadas manualmente |
| Sem bootstrap → sem viés | Alta variância (episódios inteiros) |
| Converge sob condições brandas | Explosão de estados se features ruins |
| Sem rede neural, sem GPU | Não generaliza entre estados similares |

## Referências

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, cap. 2 (bandidos multi-armados) e cap. 5 (Monte Carlo).
