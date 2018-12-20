import os
import librosa
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from joblib import dump, load
'''from pydub import AudioSegment
from pydub.utils import make_chunks'''

sample_rate = 44100
mfcc_size = 20
mfccs_array = []
mfccs_predict_array = []
train_label = []
train_data = []
predict_data = []
directory = './../Speech Dataset Uncatogorized'
saved_directory = './'
chunk_directory = './chunks/'
predict_data_directory = './../Speech Dataset/Singing/Actor_01'

svc = SVC(kernel = 'rbf', C=10, gamma=10)

def check_file_name(file):
    if 'angry' in file:
        train_label.extend(['angry'])
    elif 'fear' in file:
        train_label.extend(['fear'])
    elif 'happy' in file:
        train_label.extend(['happy'])
    elif 'neutral' in file:
        train_label.extend(['neutral'])
    elif 'sad' in file:
        train_label.extend(['sad'])
    elif 'ps' in file:
        train_label.extend(['surprised'])

def get_svc():
    pca_data = np.load(saved_directory + "/" + 'pca_data.npy')
    train_label = np.load(saved_directory + "/" + 'train_label.npy')
    print('Train SVM...')
    svc.fit(np.asarray(pca_data), np.array(train_label))
    dump(svc, 'svc.joblib')
    print 'svm saved'

def get_pca(mfccs_array):
    print 'Getting pca values'
    scaled_data = preprocessing.scale(mfccs_array)
    pca = PCA(n_components=1)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    np.save('pca_data.npy', pca_data)
    print'pca successfully saved'

def get_predict_pca(mfccs_predict_array):
    print 'Getting pca values of predict data'
    scaled_data = preprocessing.scale(mfccs_predict_array)
    pca = PCA(n_components=1)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    predict_data.extend(pca_data)
    np.save('predict_data.npy', predict_data)

def load_init_data():
    i = 0;
    print 'loading data ...'
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            mfccs_array.append([])
            check_file_name(file)
            file_path = os.path.join(directory, file)
            pcm_data, _ = librosa.load(file_path, duration=1.0)
            mfccs = librosa.feature.mfcc(pcm_data)
            reshaped_data = np.array(mfccs).ravel()
            mfccs_array[i].extend(reshaped_data)
            i += 1
    np.save('train_label.npy', np.asarray(train_label))
    get_pca(mfccs_array)

def get_saved_data():
    pca_data = np.load(saved_directory + "/" + 'pca_data.npy')
    train_label = np.load(saved_directory + "/" + 'train_label.npy')

def get_predict_data():
    i = 0;
    print 'loading predict data'
    for file in os.listdir(predict_data_directory):
        if file.endswith('.wav'):
            mfccs_predict_array.append([])
            file_path = os.path.join(predict_data_directory, file)
            pcm_data, _ = librosa.load(file_path, duration=4.0)

            mfccs = librosa.feature.mfcc(pcm_data,
                                             sample_rate,
                                             n_mfcc=mfcc_size)
            data = np.array(mfccs)
            reshaped_data = data.ravel()
            mfccs_predict_array[i].extend(reshaped_data)
            i += 1
    get_predict_pca(mfccs_predict_array)

def start_predicting():
    print 'start predicting'
    svc = load(saved_directory + "/" + 'svc.joblib')
    test_predict = svc.predict(np.load(saved_directory + "/" + 'predict_data.npy'))

    print 'saving predictions'
    df = pd.DataFrame(pd.Series(range(1, test_predict.shape[0] + 1), name='No'))
    df['Label'] = test_predict
    df.to_csv('results.csv', index=False)

'''def make_audio_chunks():
    print 'making chunks...';
    myaudio = AudioSegment.from_file("Toastmasters_Icebreaker_Speech.wav", "wav");
    chunk_length_ms = 5000;  # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms);  # Make chunks of five sec
    for i, item in enumerate(chunks):
        chunk_name = chunk_directory+"/"+"chunk{0}.wav".format(i);
        item.export(chunk_name, format="wav");
        print chunk_name + ' created!';'''


#load_init_data() #when training with new features
#get_saved_data() #pre saved pca data
#get_svc()
#make_audio_chunks()
#get_predict_data()
#start_predicting()

