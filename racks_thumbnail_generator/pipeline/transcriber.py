from abc import ABC, abstractmethod
from pathlib import Path

from google import genai
from google.genai import types as genai_types


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> str: ...


class GeminiTranscriber(Transcriber):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def transcribe(self, audio_path: Path) -> str:
        uploaded = self.client.files.upload(file=audio_path)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Content(
                    parts=[
                        genai_types.Part.from_uri(
                            file_uri=uploaded.uri,
                            mime_type=uploaded.mime_type,
                        ),
                        genai_types.Part.from_text(
                            text=(
                                "Transcribe this audio in Spanish. "
                                "Return ONLY the transcription text, no timestamps or labels."
                            )
                        ),
                    ]
                )
            ],
        )
        return response.text.strip()


class WhisperTranscriber(Transcriber):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="es",
            )
        return response.text.strip()


def get_transcriber(backend: str, google_api_key: str = "", openai_api_key: str = "") -> Transcriber:
    if backend == "whisper":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY required for Whisper transcription")
        return WhisperTranscriber(openai_api_key)
    return GeminiTranscriber(google_api_key)
