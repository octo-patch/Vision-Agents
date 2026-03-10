"""
Device enumeration and selection utilities for LocalTransport.

Provides interactive prompts for selecting audio and video devices
when running agents locally.
"""

import glob
import logging
import platform
import subprocess
from typing import Any

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    SOUNDDEVICE_AVAILABLE = False

try:
    import av  # noqa: F811

    PYAV_AVAILABLE = True
except ImportError:
    av = None  # type: ignore[assignment]
    PYAV_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_sounddevice() -> None:
    """Raise ImportError if sounddevice is not available."""
    if not SOUNDDEVICE_AVAILABLE:
        raise ImportError(
            "sounddevice is required for audio device selection. "
            "Install it with: pip install sounddevice"
        )


def _check_pyav() -> None:
    """Raise ImportError if PyAV is not available."""
    if not PYAV_AVAILABLE:
        raise ImportError(
            "PyAV is required for camera support. Install it with: pip install av"
        )


def list_audio_devices() -> None:
    """Print available audio devices for debugging."""
    if sd is None:
        print("sounddevice not installed")
        return

    print("Available audio devices:")
    print(sd.query_devices())
    print(f"\nDefault input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")


def select_audio_devices() -> tuple[int | None, int | None]:
    """Interactive prompt to select audio input and output devices.

    Returns:
        Tuple of (input_device_index, output_device_index).
        Returns None for either if using default.
    """
    _check_sounddevice()

    devices = sd.query_devices()
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]

    input_devices: list[int] = []
    print("\n" + "=" * 50)
    print("INPUT DEVICES (Microphones)")
    print("=" * 50)
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            is_default = " [DEFAULT]" if i == default_in else ""
            print(
                f"  {len(input_devices)}: {dev['name']} "
                f"({int(dev['default_samplerate'])}Hz){is_default}"
            )
            input_devices.append(i)

    output_devices: list[int] = []
    print("\n" + "=" * 50)
    print("OUTPUT DEVICES (Speakers)")
    print("=" * 50)
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            is_default = " [DEFAULT]" if i == default_out else ""
            print(
                f"  {len(output_devices)}: {dev['name']} "
                f"({int(dev['default_samplerate'])}Hz){is_default}"
            )
            output_devices.append(i)

    print("\n" + "-" * 50)

    input_device: int | None
    while True:
        try:
            choice = input(
                f"Select INPUT device [0-{len(input_devices) - 1}] (Enter for default): "
            ).strip()
            if choice == "":
                input_device = None
                print(f"  -> Using default: {devices[default_in]['name']}")
                break
            idx = int(choice)
            if 0 <= idx < len(input_devices):
                input_device = input_devices[idx]
                print(f"  -> Selected: {devices[input_device]['name']}")
                break
            print(f"  Invalid choice, enter 0-{len(input_devices) - 1} or press Enter")
        except ValueError:
            print("  Please enter a number or press Enter")

    output_device: int | None
    while True:
        try:
            choice = input(
                f"Select OUTPUT device [0-{len(output_devices) - 1}] (Enter for default): "
            ).strip()
            if choice == "":
                output_device = None
                print(f"  -> Using default: {devices[default_out]['name']}")
                break
            idx = int(choice)
            if 0 <= idx < len(output_devices):
                output_device = output_devices[idx]
                print(f"  -> Selected: {devices[output_device]['name']}")
                break
            print(f"  Invalid choice, enter 0-{len(output_devices) - 1} or press Enter")
        except ValueError:
            print("  Please enter a number or press Enter")

    print("-" * 50 + "\n")
    return input_device, output_device


def get_device_sample_rate(device_index: int | None, is_input: bool = True) -> int:
    """Get the default sample rate for a device."""
    _check_sounddevice()

    if device_index is None:
        device_index = sd.default.device[0 if is_input else 1]

    device_info = sd.query_devices(device_index)
    return int(device_info["default_samplerate"])


def list_cameras() -> list[dict[str, Any]]:
    """List available cameras on the system."""
    _check_pyav()

    cameras: list[dict[str, Any]] = []
    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stderr
            in_video_section = False
            for line in output.split("\n"):
                if "AVFoundation video devices:" in line:
                    in_video_section = True
                    continue
                if "AVFoundation audio devices:" in line:
                    break
                if in_video_section and "[AVFoundation" in line:
                    parts = line.split("]")
                    if len(parts) >= 3:
                        idx_part = parts[1].strip()
                        name_part = parts[2].strip()
                        if idx_part.startswith("["):
                            try:
                                cam_idx = int(idx_part.strip("[]"))
                                cameras.append(
                                    {
                                        "index": cam_idx,
                                        "name": name_part,
                                        "device": str(cam_idx),
                                    }
                                )
                            except ValueError:
                                pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Failed to list cameras (is ffmpeg installed?)")

    elif system == "Linux":
        video_devices = sorted(glob.glob("/dev/video*"))
        for i, dev_path in enumerate(video_devices):
            name_path = f"/sys/class/video4linux/{dev_path.split('/')[-1]}/name"
            try:
                with open(name_path) as f:
                    name = f.read().strip()
            except OSError:
                name = dev_path
            cameras.append(
                {
                    "index": i,
                    "name": name,
                    "device": dev_path,
                }
            )

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stderr
            in_video_section = False
            cam_idx = 0
            for line in output.split("\n"):
                if "DirectShow video devices" in line:
                    in_video_section = True
                    continue
                if "DirectShow audio devices" in line:
                    break
                if in_video_section and '"' in line:
                    start = line.find('"')
                    end = line.rfind('"')
                    if start != -1 and end > start:
                        name = line[start + 1 : end]
                        cameras.append(
                            {
                                "index": cam_idx,
                                "name": name,
                                "device": f'video="{name}"',
                            }
                        )
                        cam_idx += 1
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Failed to list cameras (is ffmpeg installed?)")

    return cameras


def select_video_device() -> str | None:
    """Interactive prompt to select a camera or skip.

    Returns:
        Device identifier string for the selected camera, or None if skipped.
    """
    _check_pyav()

    cameras = list_cameras()

    print("\n" + "=" * 50)
    print("VIDEO DEVICES (Cameras)")
    print("=" * 50)

    if not cameras:
        print("  No cameras detected")
        print("  (Camera support requires ffmpeg to be installed)")
        print("-" * 50 + "\n")
        return None

    for cam in cameras:
        print(f"  {cam['index']}: {cam['name']}")

    print("  n: No camera (skip)")
    print("-" * 50)

    while True:
        try:
            choice = (
                input(f"Select CAMERA [0-{len(cameras) - 1}] or 'n' to skip: ")
                .strip()
                .lower()
            )
            if choice == "n" or choice == "":
                print("  -> No camera selected")
                print("-" * 50 + "\n")
                return None
            idx = int(choice)
            if 0 <= idx < len(cameras):
                selected = cameras[idx]
                print(f"  -> Selected: {selected['name']}")
                print("-" * 50 + "\n")
                return selected["device"]
            print(f"  Invalid choice, enter 0-{len(cameras) - 1} or 'n'")
        except ValueError:
            print("  Please enter a number or 'n'")
