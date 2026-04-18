import logging
import socket
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from .utils import FitnessResult, Observation, extract_observation

logger = logging.getLogger(__name__)


class Environment:
    """Interface to the MarioAI simulator.

    Spawns the Java server as a subprocess and keeps a TCP connection open to
    it. Serialises actions sent to the server and decodes observations
    received from it.

    Attributes:
      name: the bot's name sent during the handshake.
      host: the server address.
      port: the server port.
      level_difficulty: the level difficulty. There is no upper limit, but
        values between 0 and 30 are suggested. Defaults to 0.
      level_type: 0 for overground, 1 for underground, 2 for castle, 3 for
        random. Defaults to 0.
      creatures_enabled: whether creatures are enabled. Defaults to True.
      init_mario_mode: initial Mario mode. 0 for small, 1 for large, 2 for
        large with fire. Defaults to 2.
      level_seed: the level generator seed. Defaults to 1.
      time_limit: the limit in Mario-seconds (faster than wall-clock).
        Defaults to 100.
      fast_tcp: whether to use the bit-packed ``E`` observation format.
        Defaults to False.
      visualization: whether the server should open a visualisation window.
        Defaults to True.
      custom_args: extra arguments appended to the ``reset`` command.
      fitness_values: length of the fitness tuple from the server. Defaults
        to 5.
    """

    def __init__(
        self,
        name: str = 'Unnamed agent',
        host: str = 'localhost',
        port: int = 4242,
    ) -> None:
        self.name = name
        self.host = host
        self.port = port

        self.level_difficulty = 0
        self.level_type = 0
        self.creatures_enabled = True
        self.init_mario_mode = 2
        self.level_seed = 1
        self.time_limit = 100
        self.fast_tcp = False

        self.visualization = True
        self.custom_args = ''
        self.fitness_values = 5

        self._server_process: subprocess.Popen | None = None
        self._stdout_log = None
        self._stderr_log = None
        self._tcpclient = self._run_server()

    def _check_java(self) -> None:
        try:
            logger.info('Checking if Java is installed...')
            result = subprocess.run(
                ['java', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            first_line = result.stdout.splitlines()[0].decode('ascii', errors='replace')
            logger.info('Java version: %s', first_line)
        except FileNotFoundError as exc:
            raise OSError('Java is not installed!') from exc

    def _run_server(self) -> 'TCPClient':
        self._check_java()
        source_dir = Path(__file__).resolve().parent
        server_dir = source_dir / 'server'
        log_dir = server_dir / 'tmp'
        log_dir.mkdir(parents=True, exist_ok=True)

        self._stdout_log = open(log_dir / 'server_logOut.log', 'w', encoding='utf-8')
        self._stderr_log = open(log_dir / 'server_logErr.log', 'w', encoding='utf-8')
        self._server_process = subprocess.Popen(
            ['nohup', 'java', 'ch.idsia.scenarios.MainRun', '-server', 'on'],
            cwd=server_dir,
            stdout=self._stdout_log,
            stderr=self._stderr_log,
        )

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            logger.info('Connection attempt: %d/%d', attempt, max_attempts)
            try:
                client = TCPClient(self.name, self.host, self.port)
                client.connect()
                return client
            except ConnectionRefusedError:
                if attempt == max_attempts:
                    raise
                time.sleep(5)

        raise RuntimeError('unreachable: server connection loop exited without returning')

    @property
    def connected(self) -> bool:
        return self._tcpclient.connected

    def disconnect(self) -> None:
        """Disconnect from the server and clean up resources."""
        self._tcpclient.disconnect()
        if self._server_process is not None:
            self._server_process.kill()
            self._server_process = None
        for fh in (self._stdout_log, self._stderr_log):
            if fh is not None and not fh.closed:
                fh.close()
        self._stdout_log = None
        self._stderr_log = None

    def get_sensors(self) -> Observation | FitnessResult:
        """Receive and decode the next observation from the server."""
        data = self._tcpclient.recv_data()

        if data == b'ciao':
            self._tcpclient.disconnect()
            raise OSError('server sent ciao — session closed')

        if len(data) <= 5:
            logger.warning('[ENVIRONMENT] Unexpected received data: %r', data)
            raise OSError('Unexpected received data from server')

        return extract_observation(data)

    def perform_action(self, action: Sequence[int]) -> None:
        """Serialise and send an action to the server.

        Each position of ``action`` represents a button; use 1 to press and
        0 to release::

            [backward, forward, crouch, jump, speed/bombs]

        Example::

            env.perform_action([0, 1, 0, 0, 0])  # walk right
            env.perform_action([1, 0, 0, 1, 0])  # jump left
        """
        if len(action) != 5:
            raise ValueError(f'action must have 5 elements, got {len(action)}')

        parts = []
        for value in action:
            if value == 1:
                parts.append('1')
            elif value == 0:
                parts.append('0')
            else:
                raise ValueError(f'action values must be 0 or 1, got {value!r}')

        parts.append('\r\n')
        self._tcpclient.send_data(''.join(parts).encode())

    def reset(self) -> None:
        """Reset the simulator with the currently configured attributes."""
        flags: list[tuple[str, object]] = [
            ('-maxFPS', 'on'),
            ('-ld', self.level_difficulty),
            ('-lt', self.level_type),
            ('-mm', self.init_mario_mode),
            ('-ls', self.level_seed),
            ('-tl', self.time_limit),
            ('-pw', 'off' if self.creatures_enabled else 'on'),
            ('-vis', 'on' if self.visualization else 'off'),
        ]
        if self.fast_tcp:
            flags.append(('-fastTCP', 'on'))

        argstring = ' '.join(f'{flag} {value}' for flag, value in flags)
        command = f'reset {argstring}'
        if self.custom_args:
            command = f'{command} {self.custom_args}'
        command = f'{command}\r\n'
        self._tcpclient.send_data(command.encode())


class TCPClient:
    """A simple TCP client for the MarioAI server.

    Attributes:
      name: the bot's name.
      host: the server address.
      port: the server port.
      sock: the underlying socket.
      connected: whether the client is connected.
      buffer_size: receive buffer size in bytes.
    """

    def __init__(
        self,
        name: str = '',
        host: str = 'localhost',
        port: int = 4242,
    ) -> None:
        self.name = name
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.connected = False
        self.buffer_size = 4096

    def connect(self) -> None:
        """Open the socket and perform the handshake."""
        logger.info('[TCPClient] trying to connect to %s:%d', self.host, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        logger.info('[TCPClient] connection to %s:%d succeeded', self.host, self.port)

        greeting = self.recv_data()
        logger.info('[TCPClient] greetings received: %r', greeting)

        message = f'Client: Dear Server, hello! I am {self.name}\r\n'
        self.send_data(message.encode())

        self.connected = True

    def disconnect(self) -> None:
        """Close the socket."""
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                logger.debug('[TCPClient] socket already closed')
            self.sock = None
        self.connected = False
        logger.info('[TCPClient] client disconnected')

    def recv_data(self) -> bytes:
        """Receive bytes from the server."""
        if self.sock is None:
            raise ConnectionError('socket is not connected')
        try:
            return self.sock.recv(self.buffer_size)
        except OSError as exc:
            logger.error('[TCPClient] error while receiving: %s', exc)
            raise

    # Backwards-compatible alias — older notebooks may still call recvData().
    recvData = recv_data

    def send_data(self, data: bytes) -> None:
        """Send bytes to the server."""
        if self.sock is None:
            raise ConnectionError('socket is not connected')
        try:
            self.sock.send(data)
        except OSError as exc:
            logger.error('[TCPClient] error while sending: %s', exc)
            raise OSError(f'[TCPClient] error while sending. Message: {exc}') from exc
