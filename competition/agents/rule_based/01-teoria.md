# Rule-based — Teoria

Agente **rule-based** (baseado em regras), também chamado de agente reativo ou heurístico, escolhe ações aplicando um conjunto finito de regras `SE condição ENTÃO ação`. Não há treinamento, função de valor nem parâmetros aprendidos: toda a inteligência está codificada pelo programador.

## Por que existe

É a linha de base por excelência. Antes de declarar que um agente aprendido é bom, é preciso mostrar que ele supera a heurística humana razoável. Em jogos de plataforma, uma regra tão simples quanto "pule quando houver obstáculo" já resolve boa parte dos níveis fáceis.

## Paradigma

- **Política**: determinística, codificada como uma cadeia de `if/elif/else` ou uma máquina de estados finita (FSM).
- **Memória**: normalmente stateless (decide só com o frame atual). Pode ter memória curta (ex.: "se pulei no frame anterior, não pulo agora").
- **Percepção**: um conjunto pequeno de features booleanas ou numéricas extraídas do estado bruto (ex.: "há inimigo à frente?", "há buraco à frente?").

## Pseudocódigo

```
função agir(percepção):
    se percepção.inimigo_à_frente e pode_pular:
        retorna PULAR_FRENTE
    se percepção.buraco_à_frente and pode_pular:
        retorna PULAR_FRENTE
    se percepção.muro_à_frente and pode_pular:
        retorna PULAR_FRENTE
    se percepção.muro_à_frente and não pode_pular:
        retorna RECUAR
    senão:
        retorna CORRER_FRENTE
```

## Projeto das regras

Um roteiro típico:

1. **Definir o objetivo** — em Mario, chegar à bandeira à direita.
2. **Listar os obstáculos** — inimigos, buracos, muros, canos, penhascos.
3. **Listar as ações** — andar, pular, pular+correr, recuar.
4. **Mapear condição → ação** — começar pelo caso mais perigoso (inimigo próximo) e descer para o caso default (andar).
5. **Ordenar por prioridade** — a primeira regra que casa é aplicada; regras mais críticas vêm antes.

## Trade-offs

| Vantagem | Desvantagem |
|---|---|
| Determinístico e reproduzível | Não aprende com erros |
| Zero custo computacional de treino | Exige conhecimento do domínio |
| Fácil de depurar (rastreia qual regra disparou) | Frágil fora do "envelope" previsto pelo programador |
| Baseline obrigatório antes de algoritmos aprendidos | Combinação de regras cresce exponencialmente |

## Quando funciona e quando falha

**Funciona bem** quando o espaço de estados é pequeno, as regras cobrem os casos comuns e a recompensa é simples (aqui: andar para a direita e não morrer).

**Falha** em situações não previstas — ex.: em castelos com layout incomum, com inimigos se aproximando por cima, ou com plataformas móveis. O agente segue sua regra default (correr) e morre.

## Conexão com RL

Um agente rule-based é uma política determinística *a priori*. Em RL, poderíamos ver cada regra como um par `(estado discretizado, ação)` numa Q-table populada manualmente. A única diferença é que num agente aprendido esses valores vêm de exploração + atualização de Bellman; aqui, vêm da intuição humana.

## Referências

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*, cap. 2 — Agentes reativos e agentes baseados em regras.
- Millington, I. *AI for Games*, cap. 5 — Decision trees e state machines aplicadas a jogos.
