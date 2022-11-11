import numpy as np 
import matplotlib.pyplot as plt 


import pitch,soundfile
from pydub import AudioSegment

from scipy import fft, arange

from scipy.io import wavfile
import os
import wave




    

arr_pitch = [] #data creation
gen = []
age = []
# handling data
import os
for (root,dirs,files) in os.walk('../gender_age/manual_data/'):
    for file in files:
        print(file)
        audio = os.path.join(root,file)
        audio_segment = AudioSegment.from_file(audio)
        wav_obj = wave.open(audio, 'rb')
        sample_freq = wav_obj.getframerate()
        if (wav_obj.getnchannels() == 2): #converting to single channel
            sound = AudioSegment.from_wav(audio)
            sound = sound.set_channels(1)
            sound.export(audio, format="wav")
        pit = pitch.find_pitch(audio)
        print(pit)
        det = file.split('_')
        if(det[0] == 'M'):
            gen.append([0])
        else:
            gen.append([1])
            
        arr_pitch.append([pit])

        age.append([det[1]])
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(arr_pitch, gen)

depX = []
for i in range(0,len(gen)):
    depX.append([arr_pitch[i][0],gen[i][0]])
from sklearn.tree import DecisionTreeClassifier
Decision_Tree = DecisionTreeClassifier( criterion = 'entropy' , random_state=0, max_depth = 2, min_samples_split = 3)
Decision_Tree.fit(depX, age)


test_audio = 'ren2.wav'
print('for sound test audio is ',classifier.predict([[pitch.find_pitch(test_audio)]]))
print('age prediction is',Decision_Tree.predict([[pitch.find_pitch(test_audio),0]]))


