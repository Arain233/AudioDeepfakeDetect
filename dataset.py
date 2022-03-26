import os
from PIL import Image
import librosa
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from spafe.features.mfcc import mfcc
from label import getLabel


class ASVspoofDataset(Dataset):

    def __init__(self, train=True, transform=None, target_transform=None) -> None:
        super().__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # self.root_dir = 'D:/DataSet/ASVspoof2017/ASVspoof2017_train_dev/wav'
        self.root_dir = 'D:/DataSet/ASVspoof2019/ASVspoof2019'
        if self.train:
            self.audio_path = os.path.join(self.root_dir, 'train_fig')
        else:
            self.audio_path = os.path.join(self.root_dir, 'dev_fig')
        self.audio_list = os.listdir(self.audio_path)

    def __getitem__(self, index) -> T_co:
        # sig, fs = librosa.load(self.audio_path + '/' + self.audio_list[index], sr=16000)
        # melspectrogram = librosa.feature.melspectrogram(y=sig, sr=fs, n_fft=1024, hop_length=512, n_mels=128)
        # mat = librosa.power_to_db(melspectrogram)
        # mat = mfcc(sig, fs)
        # mat = mat.astype(np.float32)
        mat = Image.open(self.audio_path + '/' + self.audio_list[index])
        if self.transform is not None:
            mat = self.transform(mat)
        label = getLabel(index, self.train)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return mat, label

    def __len__(self):
        return len(self.audio_list)
