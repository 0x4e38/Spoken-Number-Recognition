import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import random
import os
path="C:\\Users\\aswin\\Big Data\\Final Project\\data\\spoken_numbers_pcm\\"
files=os.listdir(path)
wav_file=files[random.randint(0,len(files))]

spf = wave.open(path+wav_file,'r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromiter(signal, 'Int16')
fs = spf.getframerate()

Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title('Signal Wave for'+wav_file)
plt.plot(Time,signal)
plt.show()

Y,sr=librosa.load(path+wav_file,mono=True)
mfccs=librosa.feature.mfcc(y=Y,sr=sr,n_mfcc=40)
plt.figure(figsize=(10, 4))
display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC'+wav_file)
plt.tight_layout()
plt.show()
