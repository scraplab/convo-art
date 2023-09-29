#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:35:48 2023

@author: Landry Bulls
"""

import whisper
import torch

model = whisper.load_model("small")

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

def transcribe(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio) 
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text




    
