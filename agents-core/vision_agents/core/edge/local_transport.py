"""
LocalTransport: EdgeTransport implementation for local audio/video I/O.

Uses sounddevice for microphone input and speaker output, and PyAV for
camera capture, enabling vision agents to run locally without cloud
edge infrastructure.
"""

import asyncio
import logging
import platform
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False

try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    av = None  # type: ignore[assignment]
    PYAV_AVAILABLE = False

try:
    from aiortc import VideoStreamTrack

    AIORTC_AVAILABLE = True
except ImportError:
    VideoStreamTrack = object  # type: ignore[assignment, misc]
    AIORTC_AVAILABLE = False

from getstream.video.rtc.track_util import AudioFormat, PcmData

from vision_agents.core.edge import events as edge_events
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.events import AudioReceivedEvent, TrackAddedEvent
from vision_agents.core.edge.types import Connection, Participant, TrackType, User
from vision_agents.core.events.manager import EventManager

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

logger = logging.getLogger(__name__)


def _check_sounddevice() -> None:
    """Raise ImportError if sounddevice is not available."""
    if not SOUNDDEVICE_AVAILABLE:
        raise ImportError(
            "sounddevice is required for LocalTransport. "
            "Install it with: pip install sounddevice"
        )


def _check_pyav() -> None:
    """Raise ImportError if PyAV is not available."""
    if not PYAV_AVAILABLE:
        raise ImportError(
            "PyAV is required for camera support. Install it with: pip install av"
        )


def _check_aiortc() -> None:
    """Raise ImportError if aiortc is not available."""
    if not AIORTC_AVAILABLE:
        raise ImportError(
            "aiortc is required for video track support. "
            "Install it with: pip install aiortc"
        )


def _get_camera_input_format() -> str:
    """Get the FFmpeg input format for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return "avfoundation"
    elif system == "Linux":
        return "v4l2"
    elif system == "Windows":
        return "dshow"
    else:
        raise RuntimeError(f"Unsupported platform for camera capture: {system}")


def _create_local_participant(user_id: str = "local-user") -> Participant:
    """Create a Participant representing the local user."""
    return Participant(
        original=None,
        user_id=user_id,
        id="local-session",
    )


class LocalOutputAudioTrack:
    """Audio track that plays PCM data to the speaker using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 2,
        blocksize: int = 2048,
        device: int | None = None,
    ):
        _check_sounddevice()

        self._sample_rate = sample_rate
        self._channels = channels
        self._blocksize = blocksize
        self._device = device
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=100)
        self._running = False
        self._stopped = False
        self._playback_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _playback_loop(self) -> None:
        """Dedicated thread for audio playback using blocking writes."""
        try:
            with sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                blocksize=self._blocksize,
                device=self._device,
            ) as stream:
                logger.info(
                    "Started audio output: %dHz, %d channels",
                    self._sample_rate,
                    self._channels,
                )

                while self._running:
                    try:
                        data = self._queue.get(timeout=0.1)
                        if data is None:
                            break

                        frames = len(data) // self._channels
                        audio = data.reshape(frames, self._channels)
                        stream.write(audio)

                    except queue.Empty:
                        continue

        except sd.PortAudioError:
            logger.exception("Audio playback device error")
        except ValueError:
            logger.exception("Audio data processing error")
        finally:
            logger.info("Stopped audio output")

    def start(self) -> None:
        """Start the audio output stream."""
        if self._running or self._stopped:
            return

        self._running = True
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )
        self._playback_thread.start()

    def _process_audio(self, data: PcmData) -> np.ndarray:
        """Process audio data (resample and convert) - runs in thread pool."""
        if data.sample_rate != self._sample_rate or data.channels != self._channels:
            data = data.resample(self._sample_rate, self._channels)

        samples = data.to_int16().samples

        if samples.ndim == 2:
            samples = samples.T.flatten()

        return samples

    async def write(self, data: PcmData) -> None:
        """Write PCM data to be played on the speaker."""
        if self._stopped:
            return

        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        samples = await self._loop.run_in_executor(None, self._process_audio, data)

        try:
            self._queue.put_nowait(samples)
        except queue.Full:
            logger.warning("Audio queue full, dropping samples")

    def stop(self) -> None:
        """Stop the audio output stream."""
        self._stopped = True
        self._running = False

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._playback_thread is not None:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None

    async def flush(self) -> None:
        """Clear any pending audio data."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class LocalVideoTrack(VideoStreamTrack):
    """Video track that captures from local camera using PyAV."""

    kind = "video"

    def __init__(
        self,
        device: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        _check_pyav()
        _check_aiortc()
        super().__init__()

        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._container: Any = None
        self._stream: Any = None
        self._started = False
        self._stopped = False
        self._frame_count = 0
        self._start_time: float | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _open_camera(self) -> None:
        """Open the camera device with PyAV."""
        input_format = _get_camera_input_format()
        system = platform.system()

        options: dict[str, str] = {
            "framerate": str(self._fps),
        }

        if system == "Darwin":
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
            options["pixel_format"] = "uyvy422"
        elif system == "Linux":
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
        elif system == "Windows":
            device_path = self._device
            options["video_size"] = f"{self._width}x{self._height}"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        self._container = av.open(
            device_path,
            format=input_format,
            options=options,
        )
        self._stream = self._container.streams.video[0]
        logger.info(
            "Opened camera: %s (%dx%d @ %dfps)",
            self._device,
            self._width,
            self._height,
            self._fps,
        )

    def _read_frame(self) -> Any:
        """Read a single frame from the camera (blocking)."""
        if self._container is None:
            return None

        try:
            for packet in self._container.demux(self._stream):
                for frame in packet.decode():
                    return frame
        except OSError:
            logger.warning("Error reading camera frame")
            return None
        return None

    async def recv(self) -> Any:
        """Receive the next video frame."""
        if self._stopped:
            raise RuntimeError("Track has been stopped")

        if not self._started:
            self._started = True
            self._start_time = time.time()
            self._loop = asyncio.get_running_loop()
            await self._loop.run_in_executor(None, self._open_camera)

        assert self._loop is not None
        frame = await self._loop.run_in_executor(None, self._read_frame)

        if frame is None:
            frame = av.VideoFrame(
                width=self._width, height=self._height, format="rgb24"
            )
            frame.planes[0].update(bytes(self._width * self._height * 3))

        self._frame_count += 1
        frame.pts = self._frame_count
        frame.time_base = av.Fraction(1, self._fps)  # type: ignore[attr-defined]

        return frame

    def stop(self) -> None:
        """Stop camera capture and release resources."""
        with self._lock:
            self._stopped = True
            if self._container is not None:
                try:
                    self._container.close()
                except OSError:
                    logger.warning("Error closing camera")
                self._container = None
                self._stream = None
            logger.info("Stopped camera capture")


class LocalConnection(Connection):
    """Connection wrapper for local transport."""

    def __init__(self, transport: "LocalTransport"):
        super().__init__()
        self._transport = transport
        self._participant_joined = asyncio.Event()
        self._participant_joined.set()

    def idle_since(self) -> float:
        """Local transport is never idle."""
        return 0.0

    async def wait_for_participant(self, timeout: float | None = None) -> None:
        """Local user is always present, return immediately."""
        return

    async def close(self, timeout: float = 2.0) -> None:
        """Close the local connection."""
        await self._transport._stop_audio()


class LocalTransport(EdgeTransport):
    """EdgeTransport implementation for local audio/video I/O.

    Uses sounddevice for microphone input and speaker output, and PyAV for
    camera capture. This enables running vision agents locally without cloud
    dependencies.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        input_channels: int = 1,
        output_channels: int = 2,
        blocksize: int = 1024,
        input_device: int | None = None,
        output_device: int | None = None,
        video_device: str | None = None,
        video_width: int = 640,
        video_height: int = 480,
        video_fps: int = 30,
    ):
        super().__init__()
        _check_sounddevice()

        self._sample_rate = sample_rate
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._blocksize = blocksize
        self._input_device = input_device
        self._output_device = output_device

        self._video_device = video_device
        self._video_width = video_width
        self._video_height = video_height
        self._video_fps = video_fps

        self.events = EventManager()
        self.events.register_events_from_module(edge_events)
        self._local_participant = _create_local_participant()

        self._input_stream: Any = None
        self._input_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._mic_task: asyncio.Task[None] | None = None
        self._running = False
        self._audio_track: LocalOutputAudioTrack | None = None
        self._video_track: LocalVideoTrack | None = None
        self._connection: LocalConnection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _microphone_callback_async(self, data: np.ndarray) -> None:
        """Process microphone data and emit AudioReceivedEvent."""
        samples = data.flatten().astype(np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=self._sample_rate,
            format=AudioFormat.S16,
            channels=self._input_channels,
        )
        pcm.participant = self._local_participant

        self.events.send(
            AudioReceivedEvent(
                plugin_name="local_transport",
                pcm_data=pcm,
                participant=self._local_participant,
            )
        )

    def _microphone_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Sounddevice callback for microphone input."""
        if status:
            logger.warning("Audio input status: %s", status)

        if self._running and self._loop is not None:
            self._loop.call_soon_threadsafe(self._input_queue.put_nowait, indata.copy())

    async def _microphone_loop(self) -> None:
        """Process microphone input from the queue."""
        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(self._input_queue.get(), timeout=0.1)
                    await self._microphone_callback_async(data)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.debug("Microphone loop cancelled")
            raise

    async def _start_audio(self) -> None:
        """Start microphone capture."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        self._input_stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._input_channels,
            dtype="int16",
            blocksize=self._blocksize,
            device=self._input_device,
            callback=self._microphone_callback,
        )
        self._input_stream.start()
        logger.info(
            "Started microphone: %dHz, %d channels",
            self._sample_rate,
            self._input_channels,
        )

        self._mic_task = asyncio.create_task(self._microphone_loop())

    async def _stop_audio(self) -> None:
        """Stop all audio and video streams."""
        self._running = False

        if self._mic_task is not None:
            self._mic_task.cancel()
            try:
                await self._mic_task
            except asyncio.CancelledError:
                pass
            self._mic_task = None

        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
            logger.info("Stopped microphone")

        if self._audio_track is not None:
            self._audio_track.stop()

        if self._video_track is not None:
            self._video_track.stop()
            self._video_track = None

    async def publish_tracks(  # type: ignore[override]
        self,
        audio_track: LocalOutputAudioTrack | None,
        video_track: LocalVideoTrack | None,
    ) -> None:
        """Publish the agent's media tracks locally."""
        if audio_track is not None and isinstance(audio_track, LocalOutputAudioTrack):
            audio_track.start()
            logger.info("Audio track published and started")

        if video_track is not None and isinstance(video_track, LocalVideoTrack):
            logger.info("Video track published")

    def create_audio_track(
        self, sample_rate: int = 48000, stereo: bool = True
    ) -> "LocalOutputAudioTrack":
        """Create an audio track that plays on the local speaker."""
        channels = 2 if stereo else 1
        self._audio_track = LocalOutputAudioTrack(
            sample_rate=sample_rate,
            channels=channels,
            blocksize=self._blocksize,
            device=self._output_device,
        )
        return self._audio_track

    def create_video_track(self) -> LocalVideoTrack | None:
        """Create a video track for the agent's camera input."""
        if self._video_device is None:
            logger.debug("No video device configured, skipping video track creation")
            return None

        if not PYAV_AVAILABLE or not AIORTC_AVAILABLE:
            logger.warning(
                "PyAV or aiortc not available, skipping video track creation"
            )
            return None

        self._video_track = LocalVideoTrack(
            device=self._video_device,
            width=self._video_width,
            height=self._video_height,
            fps=self._video_fps,
        )
        return self._video_track

    def add_track_subscriber(self, track_id: str) -> LocalVideoTrack | None:  # type: ignore[override]
        """Return the local camera video track if available."""
        if track_id == "local-video" and self._video_track is not None:
            return self._video_track
        return None

    async def join(
        self, agent: "Agent", call: Any = None, **kwargs: Any
    ) -> LocalConnection:  # type: ignore[override]
        """Start microphone capture and optionally camera."""
        await self._start_audio()

        if self._video_device is not None:
            video_track = self.create_video_track()
            if video_track is not None:
                self.events.send(
                    TrackAddedEvent(
                        plugin_name="local_transport",
                        track_id="local-video",
                        track_type=TrackType.VIDEO,
                        participant=self._local_participant,
                    )
                )
                logger.info("Camera video track added")

        self._connection = LocalConnection(self)
        return self._connection

    async def close(self) -> None:
        """Stop audio/video and release all resources."""
        await self._stop_audio()
        self._connection = None

    async def authenticate(self, user: User) -> None:
        pass

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        """Not supported for local transport."""
        logger.info(
            "LocalTransport does not have a demo UI. "
            "Audio is captured from your microphone and played on your speakers."
        )

    async def create_call(self, call_id: str, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("LocalTransport does not support create_call")

    async def create_conversation(  # type: ignore[override]
        self, call: Any, user: User, instructions: str
    ) -> None:
        return None

    async def send_custom_event(self, data: dict[str, Any]) -> None:
        pass
