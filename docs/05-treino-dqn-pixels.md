# Treino do agente DQN pixels+CNN

Guia rápido para rodar o agente `dqn_pixels` (9º da competição) via linha de
comando. O agente consome frames renderizados pela janela do servidor Java e
treina uma `CnnPolicy` do stable-baselines3.

## Pré-requisitos

- Python ≥ 3.10 e `venv` disponível (`apt install python3.12-venv` em
  Debian/Ubuntu, se necessário).
- Java no `PATH` (o próprio cliente sobe o servidor como subprocess).
- Sessão gráfica ativa (X11 no Linux; Windows nativo também é suportado) — a
  captura precisa enxergar a janela do jogo.

## Instalação

```bash
# 1. criar o venv
python3 -m venv .venv

# 2. instalar o pacote com os extras de captura (mss, pygetwindow / ewmh+xlib
#    no Linux, opencv, pywin32 só em Windows)
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[capture]'
```

Em Linux, os extras instalam automaticamente `ewmh` e `python-xlib` (via
marker `platform_system == 'Linux'`), usados pelo shim que substitui o
`pygetwindow` — que não suporta Linux.

## Rodando o treino

O comando abaixo treina por 10 000 passos e salva o checkpoint em
`competition/agents/dqn_pixels/dqn_pixels.zip` (padrão):

```bash
.venv/bin/python -m competition.agents.dqn_pixels.train \
  --total-timesteps 10000
```

Durante o treino, uma janela `Mario Intelligent 2.0` abre na tela — não a
minimize nem sobreponha, pois a captura lê os pixels dela.

### Smoke test rápido

Para validar a pipeline ponta-a-ponta em ~1 minuto (CUDA):

```bash
mkdir -p /tmp/mario-smoke
.venv/bin/python -m competition.agents.dqn_pixels.train \
  --total-timesteps 10000 \
  --learning-starts 500 \
  --buffer-size 5000 \
  --save-path /tmp/mario-smoke/dqn_pixels_smoke.zip
```

Resultado esperado no log: ~40 episódios, `fps ~230`, `loss` descendo e a
linha final `saved model to /tmp/mario-smoke/dqn_pixels_smoke.zip`.

### Treino sério

Para um treino "de verdade", use os defaults (200 k passos, buffer 50 k):

```bash
.venv/bin/python -m competition.agents.dqn_pixels.train \
  --total-timesteps 200000
```

A velocidade é limitada pelo servidor single-instance (porta 4242 fixa), então
espere 3–5 h de wall-clock. Redirecione o log se for deixar rodando:

```bash
nohup .venv/bin/python -m competition.agents.dqn_pixels.train \
  --total-timesteps 200000 \
  > train.log 2>&1 &
```

## Flags relevantes

| Flag                       | Default          | Descrição                                                |
|----------------------------|------------------|----------------------------------------------------------|
| `--total-timesteps`        | 200 000          | passos de ambiente                                       |
| `--learning-starts`        | 5 000            | passos de exploração pura antes do primeiro update       |
| `--buffer-size`            | 50 000           | tamanho do replay buffer                                 |
| `--batch-size`             | 32               | mini-batch do gradient step                              |
| `--learning-rate`          | 1e-4             | taxa da Adam                                             |
| `--gamma`                  | 0.99             | fator de desconto                                        |
| `--exploration-fraction`   | 0.5              | fração do treino em que o ε cai de 1.0 até o final       |
| `--exploration-final-eps`  | 0.05             | ε mínimo                                                 |
| `--target-update-interval` | 1 000            | steps entre updates do target network                    |
| `--level-difficulty`       | 3                | **≠** das seeds de avaliação                              |
| `--level-seed`             | 42               | **≠** {1001, 2042, 2077, 3013, 3099} (regras da competição) |
| `--mario-mode`             | 2                | 0=small, 1=big, 2=fire                                   |
| `--time-limit`             | 100              | Mario-segundos por episódio                              |
| `--max-fps`                | 720              | teto de FPS do servidor (mais rápido = treino mais rápido) |
| `--window-title`           | `Mario Intelligent` | substring do título da janela para a captura             |
| `--capture-backend`        | `mss`            | `mss` (default) ou `win32` (Windows, funciona minimizado)|
| `--save-path`              | `.../dqn_pixels.zip` | onde salvar o checkpoint                             |
| `--seed`                   | 42               | seed da policy                                           |

Lista completa:

```bash
.venv/bin/python -m competition.agents.dqn_pixels.train --help
```

## Troubleshooting

- **`WindowNotFoundError: no window matches 'Mario Intelligent'`**: a janela
  ainda não abriu — o servidor demora uns segundos; o cliente já tem retry.
  Se persistir, confira o título com `xdotool search --name Mario` (Linux) ou
  o gerenciador de tarefas (Windows) e passe o correto em `--window-title`.
- **Múltiplas janelas batendo o filtro**: o capture pega a maior por default;
  feche abas/terminais cujo título contenha "Mario" ou use um `--window-title`
  mais específico.
- **`AssertionError: You should use NatureCNN only with images`**: o espaço de
  observação precisa ser 3D `(H, W, C)`. Já é o caso no env shipado; se você
  escreveu um wrapper customizado, preserve o eixo de canal.
- **Servidor Java órfão** (`ch.idsia.scenarios.MainRun` rodando depois que o
  script morreu): `pkill -f ch.idsia.scenarios.MainRun`. Em uso normal o
  `env.close()` / `finally` limpa sozinho.
- **Treinando em Windows sem monitor visível**: use `--capture-backend win32`
  (`PrintWindow` funciona com janela minimizada/ocluída).
