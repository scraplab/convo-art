#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


FORMAT = pyaudio.paInt16
CHANNELS = 2
#RATE = 44100
RATE = 16000
CHUNK = 8000
THRESHOLD = 1500  # Adjust this value based on your microphone and environment
AUDIOFILE_PATH = '/Users/f004swn/Dropbox (Dartmouth College)/scrap_hack/ArtsIntegrationHack/scripts/audio_outputs'


# In[3]:


GAP = 2 # Length of silence allowed before cut
SESSION_TIME = 3 # number of mins the session can be


# In[4]:


p = pyaudio.PyAudio()


# In[5]:


#find the Scarlett
devices = []
device_index = None
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    devices.append(device_info)
    if "Scarlett" in device_info["name"] and device_info["maxInputChannels"] > 0:
        device_index = i
        break


# In[6]:


print("Waiting for speech...")


# In[7]:


stream = p.open(input_device_index=device_index,
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# In[8]:


class RecordTranscribe(threading.Thread):
    def __init__(self, rms):
        super().__init__()
        self.is_recording = True
        self.is_transcribing = False
        self.is_writing = False
        self.rms = rms
        self.text = None
        self.start_time = None
        self.stop_time = None
        self.frames = []
        self.data = None
        self.completed = False
        
    def record(self):
        self.start_time = time.time()
        print(f"{self._name} starts recording...")
        
        while True:
            self.last = time.time()
            self.data = stream.read(CHUNK, exception_on_overflow=False)
            self.frames.append(self.data)
            self.rms = audioop.rms(self.data, 2)
            
            if self.rms > THRESHOLD:
                self.last = time.time()

            elif (time.time()-self.last > GAP): # Stop recording
                print(f"LAST: {time.time()-self.last}")
                self.stop_time = time.time()
                print(f"{self._name} stops recording...")
                self.is_recording = False
                break
                
                
    def transcribe(self):
        self.is_transcribing = True
        print(f"{self._name} starts transcribing...")

        global count
        count+=1
        self.filename = os.path.join(AUDIOFILE_PATH, "output_"+str(count)+".wav")
        
        wf = wave.open(self.filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE) 
        outDat = b"".join(self.frames)
        del self.frames
        wf.writeframes(outDat)
        wf.close()
        self.text = transcribe(self.filename)
        print(f"File: {self.filename} | Text: {self.text}")
        self.is_transcribing = False
        print(f"{self._name} stops transcribing...")

            
    def write(self):
        self.is_writing = True
        newRow = pd.DataFrame.from_dict(dict(zip(OUTPUT_COLS, [[os.path.abspath(self.filename)], [self.text], [self.start_time], [self.stop_time]])))
        with writeLock:
            global output_info
            output_info = pd.concat([output_info, newRow], ignore_index=True)
            output_info.to_csv('session_output.csv')
        self.is_writing = False
            
    def run(self):
        self.record()
        self.transcribe()
        self.write()
        self.completed = True


# In[9]:


def check_recording():
    if len(thread_pool):
        if any([thread.is_recording for thread in thread_pool]):
            return True
        else:
            return False
    else:
        return False


# In[10]:


def thread_ready():
    #speaking = audioop.rms(stream.read(CHUNK, exception_on_overflow=False), 2) > THRESHOLD
    recording = check_recording()
    has_space = len(thread_pool) < NUM_THREADS
    if not recording and has_space:
        return True
    else:
        return False


# In[11]:


# init dataframe
OUTPUT_COLS = ['audio_path', 'text', 'start', 'stop']
NUM_THREADS = 3
output_info = pd.DataFrame(columns=OUTPUT_COLS)
output_info.to_csv('session_output.csv')
overall_start = time.time()

writeLock = threading.Lock()

#any_recording =  check_recording()

thread_pool = []

count = -1

while time.time() - overall_start < 60 * SESSION_TIME:
    thread_pool = [thread for thread in thread_pool if not thread.completed]
    for thread in thread_pool:
        print(thread._name)
    print(f"ACTIVE THREADS: {len(thread_pool)}")
    sound = stream.read(CHUNK, exception_on_overflow=False)
    rms = audioop.rms(sound, 2)
    # this could be an issue here where new threads keep getting recycled...?
    if rms > THRESHOLD and thread_ready():
        new_thread = RecordTranscribe(rms)
        thread_pool.append(new_thread)
        new_thread.start()


# In[169]:


len(thread_pool)


# In[170]:


a = thread_pool[0]
for i in a.__dict__.keys():
    print(i)


# In[152]:


a.is_transcribing


# In[171]:


a.rms < THRESHOLD and a.is_recording and (time.time()-a.last > GAP)


# In[164]:


a.is_recording 


# This needs to be restructured so that a new thread only gets instantiated if I start talking again while the previous thread completes. 

# In[ ]:


while True:
    
    # checking if the current session is at session time is a high-level step, above the threads. 
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


# In[ ]:




