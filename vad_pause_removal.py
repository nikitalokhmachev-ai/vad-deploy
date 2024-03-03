from utils_vad import get_speech_timestamps, read_audio
import torch
import functools
from pydub import AudioSegment 

sampling_rate = 16000
audiofile = 'recording.wav'
model = torch.jit.load("silero_vad.jit")
wav = read_audio(audiofile, sampling_rate=sampling_rate)
song = AudioSegment.from_mp3(audiofile) 
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, return_seconds=True)

songs = []
for el in speech_timestamps:
    start = int(el['start']*1000)
    end = int(el['end']*1000)
    songs.append(song[start:end])

out = functools.reduce(lambda a,b: a+b, songs)
out.export("output.wav", format="wav")