# Captura da janela do servidor

O cliente Python recebe do servidor Java apenas o estado simbólico (grid 22×22,
floats do Mario e inimigos). Para treinar agentes visuais (DQN com pixels,
imitação a partir de vídeo, dataset de gameplay) precisamos também dos frames
renderizados. Este módulo (`marioai/capture.py`) captura a janela do servidor
em tempo real e integra-se ao loop TCP existente sem modificar o servidor.

## Instalação

A captura é uma dependência opcional:

```bash
pip install -e '.[capture]'
```

Isso instala `mss` (captura), `pygetwindow` (localização da janela),
`opencv-python` (resize / grayscale) e `pywin32` (apenas em Windows).

## Uso programático

```python
from marioai.capture import GameWindowCapture

with GameWindowCapture('Mario', grayscale=True, resize=(84, 84)) as cap:
    frame = cap.capture_frame()       # np.ndarray, shape (84, 84), uint8
```

Para usar dentro do loop principal do cliente, basta passar a instância para
o `Runner`:

```python
from marioai.core import Task, Runner
from marioai.agents import RandomAgent
from marioai.capture import GameWindowCapture

capture = GameWindowCapture('Mario', grayscale=True, resize=(84, 84))
runner  = Runner(RandomAgent(), Task(), capture=capture, level_difficulty=0)
runner.run()
```

O `Experiment` interno chama `capture.capture_frame()` a cada passo, entre
`task.get_sensors()` e `agent.sense(state)`, e entrega o frame ao agente via
`agent.observe_frame(frame)`. Agentes que não sobrescrevem `observe_frame`
ignoram o frame silenciosamente — não há regressão para os agentes simbólicos
existentes.

## Configuração via variáveis de ambiente

`GameWindowCapture.from_env()` lê:

| Variável | Descrição | Exemplo |
|---|---|---|
| `MARIOAI_CAPTURE_WINDOW`   | substring do título da janela (obrigatória)        | `Mario`  |
| `MARIOAI_CAPTURE_GRAYSCALE`| `1`/`true` ativa grayscale                         | `1`      |
| `MARIOAI_CAPTURE_RESIZE`   | `WxH`                                              | `84x84`  |
| `MARIOAI_CAPTURE_BACKEND`  | `mss` (default) ou `win32`                         | `mss`    |

Se `MARIOAI_CAPTURE_WINDOW` não estiver setado, o método retorna `None` —
útil em scripts que devem rodar tanto headless quanto com captura.

## Configuração via CLI

Os comandos `mc` e `random` aceitam:

- `--capture / --no-capture`
- `--capture-window` (default: `Mario`)
- `--capture-grayscale / --no-capture-grayscale`
- `--capture-resize 84x84`
- `--capture-backend {mss,win32}`

Exemplo de smoke-test:

```bash
python -m marioai.cli random \
    --capture --capture-window Mario \
    --capture-grayscale --capture-resize 84x84
```

A cada 5 segundos, o módulo loga `[capture] XX.X fps` no `INFO`.

## Backends

### `mss` (default)

Captura a região da tela onde a janela está desenhada. É o caminho mais rápido
(~hundreds de FPS em hardware moderno). Limitação: a janela precisa estar
visível — se for minimizada ou totalmente coberta por outra janela, o frame
capturado será o que estiver "por cima".

### `win32` (Windows only, fallback)

Usa `PrintWindow` da Win32 API: pede ao processo dono da janela para se
redesenhar em um device context controlado por nós. **Funciona mesmo com a
janela minimizada ou coberta**. Mais lento que `mss` e requer `pywin32`. Use
quando precisar treinar em Windows com a janela fora da área visível
(ex.: agendamento noturno, screen-locker ativo).

Em qualquer SO que não seja Windows, escolher `backend='win32'` levanta
`CaptureBackendError` ao chamar `start()`.

## Resiliência

- A bbox da janela é re-lida a cada 30 frames — arrastar a janela durante o
  jogo não quebra a captura.
- Se a captura falhar (ex.: a janela foi fechada momentaneamente), o módulo
  tenta `find_window()` mais 3 vezes com backoff linear (50/100/150 ms). Se
  falhar nas três, levanta `WindowDisappearedError` — o `Experiment` propaga
  o erro para o caller.
- O `Experiment` envolve a chamada a `capture_frame()` em `try/except` para
  garantir que falhas de captura nunca derrubem o loop TCP — em caso de
  exceção, o agente recebe `frame=None` no `observe_frame`.

## Notas de plataforma

- **Wayland (Linux)**: `pygetwindow` depende de Xlib; em sessões Wayland puras
  ele não enxerga as janelas. Rode o servidor Java sob XWayland (já é o
  default na maioria dos compositores) ou logue numa sessão X11.
- **Multi-monitor**: `mss` lida corretamente com múltiplos monitores —
  `find_window()` retorna coordenadas absolutas que `mss.grab` aceita.
- **Janelas com mesmo título**: passe `window_index` para escolher entre as
  matches (ordenadas pela maior área primeiro).

## Quando NÃO usar a captura

- Treinos puramente simbólicos (DQN sobre o grid 22×22, agentes tabulares,
  rule-based) não precisam dos pixels — ativar a captura só adiciona overhead.
- CI / testes unitários: nenhum teste precisa de display real; tudo é mockado
  via `monkeypatch` dos loaders `_load_mss` / `_load_pygetwindow`.
