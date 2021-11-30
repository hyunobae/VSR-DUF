import tensorflow as tf
import numpy as np
import tensorflow.keras
from utils import FR_52L, load_datasets, gkern, DownSample

x_train = load_datasets("dataset/train/G")
#x_train = x_train.reshape(x_train, [16, 7, -1, -1, 3])
h = gkern(13, 1.6)  # 13 and 1.6 for x4
h = h[:, :, np.newaxis, np.newaxis].astype(np.float32)  # 가우시안 필터 생성

# with tf.device('gpu:1'):
H = x_train  # H: HR inputs
L_ = DownSample(H, h, 4)  # 필터 씌우고 downsample한 LR 생성 / LR frames
L = L_[:, :, 2:-2, 2:-2, :]
mf, mr = FR_52L(L, True)