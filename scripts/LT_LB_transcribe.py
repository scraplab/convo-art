#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:35:48 2023

@author: Landry Bulls
"""

import whisperx
import whisper
import librosa
import pandas as pd
import torch
import glob
import joblib
from tqdm import tqdm
import os

model = whisper.load_model("small")

def transcribe(file):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Loading file...")
    audio, sr = librosa.load(file)
    print("Resampling...")
    audio_resamp = librosa.resample(audio, sr, whisper.audio.SAMPLE_RATE)
    print("Transcribing...")
    result = model.transcribe(audio_resamp)
    print("Aligning...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, file, device)
    return result_aligned

files = glob.glob('../audio/*.wav')
results = []

for f in tqdm(files):
    result = transcribe(f)
    joblib.dump(result, os.path.join("../transcripts/", os.path.basename(f)[:-4]+".pkl"))
    results.append(result)
    
    
