import whisper
import os
from tqdm import tqdm
import torch
import pickle5 as pickle
import argparse
import json

#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f = open("test_dict.json")
details = json.load(f)
f.close()
wavs = details["path"]
model = whisper.load_model("large-v3", download_root = "data1/soumyad/whisper-models/")
model = model.to(device)
transcript_dict = {"path":[], "label":[], "text":[], "valence":[], "arousal":[], "dominance":[]}
for i, wav in enumerate(tqdm(wavs)):
    audio = whisper.load_audio(wav)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    transcript_dict["path"].append(wav)
    transcript_dict["text"].append(result.text)

with open("test_transcripts.json", "w") as f:
    json.dump(transcript_dict, f)


