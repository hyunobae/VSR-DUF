## -*- coding: utf-8 -*-
import random

import tensorflow.compat.v1 as tf
import numpy as np
import time
import sys, os
import math
from utils import Conv3D, BatchNorm, Huber, LoadImage
import glob
from tensorflow.keras.utils import Sequence

tf.disable_v2_behavior()

VERSION = 123
MODEL = 'DF'
nb_batch = 8

#
# def load_datasets(path):
#     dir = os.listdir(path)
#     dir.sort()
#     frames = []
#     for i in range(len(dir)):
#         dir_frame = glob.glob(path + '/' + dir[i] + '/*.png')
#         for f in dir_frame:
#             frames.append(LoadImage(f))
#             print(f"{f} appended")
#
#     frames = np.asarray(frames)
#     print('func:' + frames.shape)
#     return frames

def readdata(dir, idx):
    frames = []
    dir_frame = os.listdir(dir)
    for i in range(7):
        frames.append(LoadImage(dir+'hr'+str(idx+i)+'.png'))
        # print(f"{i} appended")

    frames = np.asarray(frames)
    # print(frames.shape)
    return frames


# x = load_datasets('D:/compressed_dataset/train')
x = 'D:/VSR-DUF/dataset/train/G'

# val = load_datasets('./dataset/val/G')

class CustomDataloader(Sequence):
    def __init__(self, x_set, batch_size, shuffle=False):
        self.x = x_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        self.prev = 0
        self.videolist = os.listdir(self.x)

    def __len__(self):
        # return int(int(len(self.x) / 7) / self.batch_size)
        return 1

    def __getitem__(self, idx):
        frames = []
        for i in range(self.batch_size):
            idx_video = random.randint(0, len(self.videolist)-1)
            cnt_frame = len(os.listdir(self.x+'/'+self.videolist[idx_video]))
            idx_frame = random.randint(0, cnt_frame-8)
            hrdir = self.x + '/' + self.videolist[idx_video] + '/'

            for j in range(7):
                frames.append(LoadImage(hrdir + 'hr' + str(idx_frame+j) + '.png'))
            # frames.extend(readdata(hrdir, idx_frame))

        frames = np.asarray(frames)

        # frames = np.squeeze(frames, 0)
        frames = np.reshape(frames, (self.batch_size, 7, 960, 540, 3))
        print(frames)

        return frames
        #
        #
        # self.prev = (7 * idx) * self.batch_size
        # if idx == 0:
        #     self.next = self.batch_size * (idx + 7)
        # else:
        #     self.next = (14 * idx) * self.batch_size + 1
        #
        # indices = self.indices[self.prev:self.next]  # 전체 데이터의 index 배
        # print(self.prev, self.next)
        # batch_x = [self.x[i] for i in indices]
        # batch_x = np.asarray(batch_x)
        # print(batch_x.shape)
        #
        # batch_x = np.reshape(batch_x, (-1, 7, 960, 540, 3))
        #
        # return batch_x  # (batch, 7, height, width, color)


train = CustomDataloader(x, batch_size=nb_batch)

# for e in range(3):
#     print(e)
#     for x in train:
#         print(x.shape)

# val = CustomDataloader(val, batch_size=nb_batch)

def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], ds_x[4]])

    y = tf.depth_to_space(x, block_size)

    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x


def DynFilter3D(x, F, filter_size):
    '''
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    '''

    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)),
                                       (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    filter_localexpand = tf.Variable(filter_localexpand_np, trainable=False, dtype='float32',
                                     name='filter_localexpand')  # 5,5,5,125
    x = tf.transpose(x, perm=[0, 2, 3, 1])  # b, h, w, t
    x_localexpand = tf.nn.conv2d(x, filter_localexpand, [1, 1, 1, 1], 'SAME')  # batch, 32,32,125
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, 32,32,1,125
    x = tf.matmul(x_localexpand, F)  # b, 32,32,1,16
    x = tf.squeeze(x, axis=3)  # b, 32,32,16

    return x


def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


h = gkern(kernlen=13, nsig=1.6)
h = h[:, :, np.newaxis, np.newaxis].astype(np.float32)


def DownSample(x, h, scale=4):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0] * ds_x[1], ds_x[2], ds_x[3], 3])

    ## Reflect padding
    W = tf.constant(h)

    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    depthwise_F = tf.tile(W, [1, 1, 3, 1])
    y = tf.nn.depthwise_conv2d(tf.pad(x, pad_array, mode='REFLECT'), depthwise_F, [1, scale, scale, 1], 'VALID')

    ds_y = tf.shape(y)
    y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return y


stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
sp5 = [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]]


def FR(x):
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID',
               name='conv1')  # b, 3,32,32,256

    #    x = tf.squeeze(x, 1) # b, 128,128,64
    F = 64
    G = 32
    for r in range(3):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x, t], 4)
        F += G
    for r in range(3, 6):
        t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
        t = tf.nn.relu(t)
        t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

        t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
        t = tf.nn.relu(t)
        t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
                   name='Rconv' + str(r + 1) + 'b')

        x = tf.concat([x[:, 1:-1], t], 4)
        F += G

    #    x = x[:,1:-1]

    x = BatchNorm(x, is_train, name='fbn1')
    x = tf.nn.relu(x)
    x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')

    x = tf.nn.relu(x)

    r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
    r = tf.nn.relu(r)
    r = Conv3D(r, [1, 1, 1, 256, 3 * 16], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

    f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')  # Fx: b,3,32,32,1200
    f = tf.nn.relu(f)
    f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * 16], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

    ds_f = tf.shape(f)
    f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])

    f = tf.nn.softmax(f, dim=4)

    return f, r


def G(x):
    # x : b,5,32,32,3
    Fx, Rx = FR(x)  # Fx: b,2,32,32,75,16
    Rx = depth_to_space_3D(Rx, 4)
    x_c = []
    x_f = []
    for f in range(1):
        for c in range(3):
            t = DynFilter3D(x[:, f + 3:f + 4, :, :, c], Fx[:, f, :, :, :, :], [1, 5, 5])  # b,32,32,16
            t = tf.depth_to_space(t, 4)  # b,128,128,1
            t = tf.squeeze(t, axis=3)  # b,128,128
            x_c += [t]
        x_f += [tf.stack(x_c, axis=3)]  # b,128,128,3
        x_c = []
    x = tf.stack(x_f, axis=1)  # b,2,128,128,3
    x += Rx

    return x, tf.reshape(Fx[:, 0, 16, 16, :, 0], [-1, 1, 5, 5, 1]), Rx


# with tf.device('gpu:1'): # placeholder는 담는 변수, 변수를 사용하는 부분을 run 해주면 된다.
H = tf.placeholder(tf.float32, shape=[None, 7, None, None, 3])
L_ = DownSample(H, h, 4)
L = L_[:, :, 2:-2, 2:-2, :]

is_train = tf.placeholder(tf.bool, shape=[])  # Phase ,scalar
lr_G = tf.placeholder(tf.float32, shape=[])

with tf.variable_scope('G') as scope:
    GL, Fx, Rx = G(L)

loss_M = Huber(H[:, 3, 24:-24, 24:-24, :], GL[:, 0, 16:-16, 16:-16, :], 0.01)
loss_M *= 1.

# Total loss
loss_G = loss_M

trainable_G = [v for v in tf.trainable_variables() if v.name.startswith('G/')]
for v in trainable_G:
    print(v)

train_G = tf.train.AdamOptimizer(lr_G).minimize(loss_G, var_list=trainable_G)

params_G = [v for v in tf.global_variables() if v.name.startswith('G/')]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    resume = 0
    start_flag = True

    curr_lr_G = 0.001
    dT = 0.
    rT = 0.
    for i in range(resume, 220001):
        # if i % 1000 == 0 and i != 0:

        #             # VSRval4
        #             valScenes = ['coastguard', 'foreman', 'garden', 'husky']
        #             n = 0
        #             vid4 = 0
        #             val_Gs = []
        #             for valScene in valScenes:
        #                 fnames = glob.glob('./Val_mat/VSRVal4/'+valScene+'/*.png')
        #                 fnames.sort()
        #
        #                 val_Hs = []
        #                 for fname in fnames:
        #                     val_Hs.append(_load_img_array(fname))
        #
        #                 val_H = np.asarray(val_Hs)
        #                 val_G = np.empty_like(val_H)
        #
        #                 val_H_ = np.lib.pad(val_H, pad_width=((3,3),(0,0),(0,0),(0,0)), mode = 'constant')
        #                 val_H_ = np.lib.pad(val_H_, pad_width=((0,0),(8,8),(8,8),(0,0)), mode = 'reflect')
        #
        #                 for f in range(0,val_H_.shape[0]-6):
        #                     in_H = val_H_[f:f+7] #select 4 frames
        #                     in_H = in_H[np.newaxis,:,:,:,:]
        #
        #                     out_G = sess.run(GL, feed_dict={H: in_H, is_train: False})
        #                     out_G = np.clip(out_G[0,0], 0. , 1.)
        #                     plt.imsave('validation/'+MODEL+'/v'+str(VERSION)+'_i'+ str(i) + '_Vid'+str(n)+'_f'+str(f)+".png", (out_G+1.)/2., vmin=0, vmax=1)
        #                     val_G[f] = out_G

        t = time.time()
        for f in train:
            batch_H = f  # 배치 데이터로더 여기에 연결시키면 되겠다.

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

        if (i % 10000 == 0) and i != 0:
            saver.save(sess, save_path='save/' + str(i) + '/.h5')
            # SaveParams(sess, [params_G], out_file='checkpoints/'+MODEL+'/v'+str(VERSION)+'_i{:d}.h5'.format(i))

        if i % 100000 == 0 and i != 0:
            curr_lr_G = np.maximum(0.1 * curr_lr_G, 0.00001)
            print('lr:', curr_lr_G)

    print('Done')