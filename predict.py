# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
import functools
from cog import BasePredictor, Input, Path
from utils_vad import get_speech_timestamps, read_audio
from pydub import AudioSegment

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = torch.jit.load("silero_vad.jit")

    def predict(
        self,
        input_audio: Path = Input(
            description="Input audio file with speech.",
            default=None,
        ),
        sampling_rate: int = Input(
            description="Sampling rate.",
            default=16000,
            choices=[16000, 8000],
        ),
        out_format: str = Input(
            description="Output format.",
            default="mp3",
            choices=["mp3", "wav"],
        )
    ) -> Path:
        data = read_audio(input_audio, sampling_rate=sampling_rate)
        song = AudioSegment.from_file(input_audio)
        speech_timestamps = get_speech_timestamps(data, self.model, sampling_rate=sampling_rate, return_seconds=True)

        songs = []
        for el in speech_timestamps:
            start = int(el['start']*1000)
            end = int(el['end']*1000)
            songs.append(song[start:end])

        out = functools.reduce(lambda a, b: a+b, songs)
        out.export(f"output.{out_format}", format=out_format)

        return Path(f"output.{out_format}")
