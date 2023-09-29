#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 21:39:48 2023

@author: f004swn
"""

import time

def wait(idx):
    #idx = 0
    bars = [['|    '],
            [' |   '],
            ['  |  '],
            ['   | '],
            ['    |'],
            ['     ']]
    
    print(bars[idx % len(bars)][0], end="\r")
    idx += 1
    time.sleep(0.1)