import tempfile
from pathlib import Path

import ffmpeg


def extract_audio(video_path: Path) -> Path:
    """Extract audio from video to a temporary WAV file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    output_path = Path(tmp.name)

    (
        ffmpeg
        .input(str(video_path))
        .output(str(output_path), acodec="pcm_s16le", ar=16000, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path
