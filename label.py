import pandas as pd

# train_label = pd.read_csv('D:/DataSet/ASVspoof2017/ASVspoof2017_train_dev/protocol/ASVspoof2017_train.trn',
#                           sep=' ', header=None)
# dev_label = pd.read_csv('D:/DataSet/ASVspoof2017/ASVspoof2017_train_dev/protocol/ASVspoof2017_dev.trl',
#                         sep=' ', header=None)

train_label = pd.read_csv('D:/DataSet/ASVspoof2019/ASVspoof2019.LA.cm.train.trn.csv', header=None)
dev_label = pd.read_csv('D:/DataSet/ASVspoof2019/ASVspoof2019.LA.cm.dev.trl.csv', header=None)


def getLabel(audio_index, train=True):
    if train:
        label = train_label.iloc[audio_index, 3]
    else:
        label = dev_label.iloc[audio_index, 3]
    # convert to class
    # TTS
    if label == 'A01':
        label = 1
    elif label == 'A02':
        label = 1
    # VC
    elif label == 'A03':
        label = 2
    elif label == 'A04':
        label = 2
    elif label == 'A05':
        label = 2
    elif label == 'A06':
        label = 2
    # REAL
    else:
        label = 0
    return label
