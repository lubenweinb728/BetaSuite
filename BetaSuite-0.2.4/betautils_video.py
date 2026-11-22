import json
import subprocess


def probe_video(filepath):
    command = [
        'ffprobe', '-v', 'error', '-show_streams', '-show_format',
        '-of', 'json', str(filepath)
    ]
    result = subprocess.run(command, capture_output=True, check=True, timeout=60)
    return json.loads(result.stdout)


def stream_start(stream, default=0.0):
    try:
        return float(stream['start_time'])
    except (KeyError, ValueError, TypeError):
        return default
