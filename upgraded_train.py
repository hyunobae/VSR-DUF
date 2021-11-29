## -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time

import sys

from utils import AVG_PSNR
from nets import FR_52L as FR
from utils import *
from tensorflow import keras

from scipy.io import loadmat
import glob
import scipy

VERSION = 123
MODEL = 'DF'
nb_batch = 16

x_train_path = './dataset/train/G'
# y_train_path = './dataset/train/G'

x_train = load_datasets(x_train_path)
# y_train = load_datasets(y_train_path)

x_train = x_train.reshape(x_train, [nb_batch, 7, -1, -1, 3])  # [batch, 7, w, h, 3]

stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
sp5 = [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]]

h = gkern(13, 1.6)  # 13 and 1.6 for x4
h = h[:, :, np.newaxis, np.newaxis].astype(np.float32)  # 가우시안 필터 생성

# with tf.device('gpu:1'):
H = x_train  # H: HR inputs
L_ = DownSample(H, h, 4)  # 필터 씌우고 downsample한 LR 생성 / LR frames
L = L_[:, :, 2:-2, 2:-2, :]

is_train = True  # Phase ,scalar
lr_G = 0.001  # 초기 learning rate

# GL, Fx, Rx = G(L) # output HR frame, filter output, residual output

# loss_M = Huber(H[:,3,24:-24,24:-24,:], GL[:,0,16:-16,16:-16,:], 0.01)
# loss_M *= 1.

# Total loss
# loss_G = loss_M


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '1'


@tf.function
def forward(x):
    GL, Fx, Rx = G(x)
    loss_M = Huber(H[:, 3, 24:-24, 24:-24, :], GL[:, 0, 16:-16, 16:-16, :], 0.01)
    loss_M *= 1.
    loss_G = loss_M

    return GL, loss_G


with tf.compat.v1.Session(config=config) as sess:
    tf.compat.v1.global_variables_initializer().run()
    resume = 0
    start_flag = True

    # restore v13
    #    LoadParams(sess, [params_G], in_file='checkpoints/DF/v82_i190000.h5')

    curr_lr_G = 0.001
    best_vid4 = 25 * 4
    dT = 0.
    rT = 0.
    for i in range(resume, 220001):
        sess.run(train_G, feed_dict={H: x_train})
        if i % 1000 == 0 and i != 0:

            # VSRval4
            valScenes = ['coastguard', 'foreman', 'garden', 'husky']
            n = 0
            vid4 = 0
            val_Gs = []
            for valScene in valScenes:
                fnames = glob.glob('./Val_mat/VSRVal4/' + valScene + '/*.png')
                fnames.sort()

                val_Hs = []
                for fname in fnames:
                    val_Hs.append(_load_img_array(fname))

                val_H = np.asarray(val_Hs)
                val_G = np.empty_like(val_H)

                val_H_ = np.lib.pad(val_H, pad_width=((3, 3), (0, 0), (0, 0), (0, 0)), mode='constant')
                val_H_ = np.lib.pad(val_H_, pad_width=((0, 0), (8, 8), (8, 8), (0, 0)), mode='reflect')

                for f in range(0, val_H_.shape[0] - 6):
                    in_H = val_H_[f:f + 7]  # select 4 frames
                    in_H = in_H[np.newaxis, :, :, :, :]

                    out_G = sess.run(GL, feed_dict={H: in_H, is_train: False})
                    out_G = np.clip(out_G[0, 0], 0., 1.)
                    #                    plt.imsave('validation/'+MODEL+'/v'+str(VERSION)+'_i'+ str(i) + '_Vid'+str(n)+'_f'+str(f)+".png", (out_G+1.)/2., vmin=0, vmax=1)
                    val_G[f] = out_G

                _psnr = AVG_PSNR(((val_H) * 255).astype(np.uint8) / 255.0, ((val_G) * 255).astype(np.uint8) / 255.0,
                                 vmin=0, vmax=1, t_border=1, sp_border=16)

                n += 1
                vid4 += _psnr
                val_Gs += [val_G]
                print('#{}: {}'.format(n, _psnr))
            print('Val4: {}'.format(vid4 / 4.))

            if vid4 > best_vid4:
                print('Saving the Best')
                n = 0
                best_vid4 = vid4
                for vid in val_Gs:
                    f = 0
                    for frame in vid:
                        plt.imsave(
                            'validation/' + MODEL + '/v' + str(VERSION) + '_i' + str(i) + '_Vid' + str(n) + '_f' + str(
                                f) + ".png", (frame), vmin=0, vmax=1)
                        f += 1
                    n += 1
                SaveParams(sess, [params_G],
                           out_file='checkpoints/' + MODEL + '/Best_v' + str(VERSION) + '_i{:d}.h5'.format(i))

        t = time.time()
        batch_H = Iter_H.dequeue()
        dT += time.time() - t

        t = time.time()
        l_M, l_G, _ = sess.run([loss_M, loss_G, train_G], feed_dict={H: batch_H, is_train: True, lr_G: curr_lr_G})
        rT += time.time() - t

        if i % 100 == 0:
            print('I : {:4d} | M: {: 4.3e} | G: {: 4.3e} | dT: {: 4.3f} | rT: {: 4.3f} s'.format(i, l_M, l_G, dT / 100.,
                                                                                                 rT / 100.))
            dT = 0.
            rT = 0.

            batch_G, batch_R = sess.run([GL, Rx], feed_dict={H: batch_H, is_train: False})
            plot_images(batch_H[:8, 1, 24:-24, 24:-24], batch_R[:8, 0, 16:-16, 16:-16], batch_G[:8, 0, 16:-16, 16:-16],
                        4, [96, 96], iter=i, is_show=False, prefix='validation/DF_show/v' + str(VERSION), postfix='_f0')

        if (i % 10000 == 0) and i != 0:
            SaveParams(sess, [params_G], out_file='checkpoints/' + MODEL + '/v' + str(VERSION) + '_i{:d}.h5'.format(i))

        if i % 100000 == 0 and i != 0:
            curr_lr_G = np.maximum(0.1 * curr_lr_G, 0.00001)
            print('lr:', curr_lr_G)

    print('Done')
