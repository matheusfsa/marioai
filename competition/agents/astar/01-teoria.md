# A* — Teoria

**A*** (pronuncia-se "A-estrela") é um algoritmo de busca em grafos informado (heurístico), proposto por Hart, Nilsson e Raphael em 1968. Encontra o **caminho de menor custo** entre um nó inicial e um nó objetivo combinando o custo acumulado `g(n)` com uma estimativa admissível `h(n)` do custo restante.

## Intuição

Busca em largura (BFS) expande uniformemente em todas as direções; busca gulosa ignora custo passado e segue só a heurística. A* faz o melhor de dois mundos: expande sempre o nó com menor `f(n) = g(n) + h(n)` — priorizando caminhos que somam progresso real + promessa de proximidade do objetivo.

## Definições

- `g(n)` — custo real acumulado do início até o nó `n`.
- `h(n)` — heurística: estimativa do custo de `n` até o objetivo. Deve ser **admissível** (nunca superestima) para garantir otimalidade.
- `f(n) = g(n) + h(n)` — custo total estimado do caminho que passa por `n`.

Se `h ≡ 0`, A* degenera em Dijkstra. Se `h` é admissível e consistente (satisfaz desigualdade triangular), A* retorna o caminho ótimo e nunca reexpande nós.

## Pseudocódigo

```
função A*(inicio, objetivo):
    abertos  = fila de prioridade ordenada por f(n)
    fechados = conjunto vazio
    g[inicio] = 0
    f[inicio] = h(inicio)
    empilha inicio em abertos

    enquanto abertos não vazio:
        n = abertos.pop_min()  // menor f(n)
        se n == objetivo:
            retorna reconstrói_caminho(n)
        fechados.adiciona(n)

        para cada vizinho v de n:
            se v em fechados: continua
            g_tentativo = g[n] + custo(n, v)
            se g_tentativo < g[v] (ou v não em abertos):
                pai[v] = n
                g[v]   = g_tentativo
                f[v]   = g[v] + h(v)
                abertos.insere_ou_atualiza(v)

    retorna FALHA
```

## Heurísticas comuns em grids

| Métrica | Quando usar |
|---|---|
| **Manhattan** `|dx| + |dy|` | Movimento restrito a 4 direções |
| **Chebyshev** `max(|dx|, |dy|)` | Movimento em 8 direções, custo uniforme |
| **Octile** `max(|dx|, |dy|) + (√2−1)·min(|dx|, |dy|)` | 8 direções, diagonal custa √2 |
| **Euclidiana** `√(dx² + dy²)` | Qualquer direção, custo geométrico |

Todas são admissíveis em grids sem descontos.

## Complexidade

- Tempo: `O(b^d)` no pior caso (`b` = fator de ramificação, `d` = profundidade). Na prática, fortemente dependente da qualidade de `h`.
- Espaço: também `O(b^d)` porque precisa manter o mapa de explorados.

## A* em ambientes dinâmicos

Mario não é um mundo estático: inimigos andam, plataformas podem se mover, o próprio Mario cai por gravidade. Três estratégias para adaptar A*:

1. **Replanejamento periódico**: recalcula o caminho a cada N frames.
2. **D\* Lite** (Koenig & Likhachev, 2002): atualiza incrementalmente o caminho quando o mapa muda.
3. **Custos dinâmicos**: inflaciona custos de células perigosas (próximas a inimigos) sem mudar a topologia.

## Pulos e gravidade

Em jogos de plataforma, a modelagem da vizinhança de um nó tem que refletir a física: de uma célula `(x, y)` no chão pode-se andar para `(x±1, y)` e também **saltar** em arcos parabólicos que cobrem N células à frente e M células acima. Uma modelagem grosseira trata "pular" como uma aresta especial que conecta `(x, y)` a `(x+k, y-h)` com custo proporcional à altura/distância do pulo.

## Trade-offs

| Vantagem | Desvantagem |
|---|---|
| Caminho ótimo garantido (h admissível) | Precisa de um modelo do ambiente |
| Determinístico | Replanejar custa CPU a cada frame |
| Não precisa de treino | Não generaliza: todo novo mapa = nova busca |
| Fácil de depurar | Modelagem de pulos é delicada |

## Referências

- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A formal basis for the heuristic determination of minimum cost paths*. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100–107.
- Koenig, S., & Likhachev, M. (2002). *D\* Lite*. AAAI.
- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*, cap. 3 — busca informada.
