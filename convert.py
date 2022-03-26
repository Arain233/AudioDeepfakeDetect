import os
import librosa
from matplotlib import pyplot as plt
from spafe.features.mfcc import mfcc

train_path = "D:/DataSet/ASVspoof2019/train/"
dev_path = "D:/DataSet/ASVspoof2019/dev/"
eval_path = "D:/DataSet/ASVspoof2019/eval/"

train_spath = "D:/DataSet/ASVspoof2019/train_fig/"
dev_spath = "D:/DataSet/ASVspoof2019/dev_fig/"
eval_spath = "D:/DataSet/ASVspoof2019/eval_fig/"

train_list = os.listdir(train_path)
dev_list = os.listdir(dev_path)
eval_list = os.listdir(eval_path)
audio_list = train_list + dev_list + eval_list

for i in range(len(audio_list)):
    if i == 0:
        read_path = train_path
        save_path = train_spath
    elif i == len(train_list):
        read_path = dev_path
        save_path = dev_spath
    elif i == len(train_list) + len(dev_list):
        read_path = eval_path
        save_path = eval_spath

    # audio convert to mfcc
    sig, fs = librosa.load(read_path + audio_list[i], sr=16000)
    mfccs = mfcc(sig, fs)

    # create fig
    plt.axis('off')
    plt.imshow(mfccs.T, origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
    plt.savefig(save_path + audio_list[i] + ".jpg", bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print("saved fig " + str(i))
