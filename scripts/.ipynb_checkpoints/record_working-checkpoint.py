#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pyaudio
import wave
import audioop
import os
import pandas as pd
import numpy as np
import time
import whisper
from transcriber import transcribe
import threading


# In[27]:


FORMAT = pyaudio.paInt16
CHANNELS = 2
#RATE = 44100
RATE = 16000
CHUNK = 10000
THRESHOLD = 1500  # Adjust this value based on your microphone and environment
AUDIOFILE_PATH = '/Users/f004swn/Dropbox (Dartmouth College)/scrap_hack/ArtsIntegrationHack/scripts/audio_outputs'


# In[28]:


GAP = 2 # Length of silence allowed before cut
SESSION_TIME = 3 # number of mins the session can be


# In[29]:


p = pyaudio.PyAudio()


# In[30]:


#find the Scarlett
devices = []
device_index = None
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    devices.append(device_info)
    if "Scarlett" in device_info["name"] and device_info["maxInputChannels"] > 0:
        device_index = i
        break


# In[31]:


print("Waiting for speech...")


# In[32]:


stream = p.open(input_device_index=device_index,
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# In[36]:


# init dataframe
output_info = pd.DataFrame(columns=['audio_path', 'text'])
output_info.to_csv('session_output.csv')
overall_start = time.time()

frames = []
recording = False
count = -1

while True:
    if time.time() - overall_start > 60*SESSION_TIME:
        print("Ending session")
        break
    data = stream.read(CHUNK, exception_on_overflow=False)
    rms = audioop.rms(data, 2)  # Calculate Root Mean Square (RMS) volume
    if rms > THRESHOLD and not recording:  # Start recording
        print("Recording started")
        recording = True
        last = time.time()
    elif rms < THRESHOLD and recording and (time.time()-last > GAP): # Stop recording
        count+=1
        print("Recording stopped")
        recording = False
        filename = os.path.join(AUDIOFILE_PATH, "output_"+str(count)+".wav")
        wf = wave.open(filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE) 
        outDat = b"".join(frames)
        frames = []
        wf.writeframes(outDat)
        wf.close()
        print("Transcribing...")
        text = transcribe(filename)
        print(text)
        newRow = pd.DataFrame.from_dict(dict(zip(['audio_path', 'text'], [[os.path.abspath(filename)], [text]])))
        output_info = pd.concat([output_info, newRow], ignore_index=True)
        output_info.to_csv('session_output.csv')
        
    if recording:
        frames.append(data)
        if rms > THRESHOLD:
            last = time.time()
            
# Need to add another condition check so that the loop breaks after a given period of time or 


# This needs to be restructured so that a new thread only gets instantiated if I start talking again while the previous thread completes. 

# In[ ]:




