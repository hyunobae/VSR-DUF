## -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time

import sys
sys.path.insert(0,'/hdd2/repository/CIP-Lab/sr/')
from layers import UpSample2DMatlab, DownSample2DMatlab, Conv3D, Standardize, Atrous_Conv2D, Blur2D, Atrous_Conv2D_T, PeriodicShuffle, Dense, Conv2D, Conv2D_T, BatchNorm, DownSample2D, AddGaussianNoise, NIN_block, LeakyReLU, SubstractMean
from loss import MSE, MAE, Huber
from sr_utils import _load_img_array
from sr_utils import *
from utils import AVG_PSNR
from nets import FR_52L as FR


from scipy.io import loadmat
import glob
import scipy

import tensorflow.contrib.slim as slim

VERSION = 123
MODEL = 'DF'
nb_batch = 16

Iter_H = GeneratorEnqueuer(DirectoryIterator_VSR_FromPickle('/hdd2/datasets/VSR/',
                                 listfile='vsr_traindata_filelist.pickle',
                                 datafile='./vsr_traindata_144_nframe31_cpi2_batch16_i10000_.pickle',
                                 total_samples=160000,
                                 target_size=144, 
                                 nframe = 7,
                                 maxbframe = 3,
                                 crop_per_image=2,
                                 out_batch_size=nb_batch, 
                                 shuffle=False))
Iter_H.start(max_q_size=16, workers=4)



def plot_images(true, rx, pred, num_image, image_shape, iter=0, is_show=False, prefix='', postfix='', no_pp=False):
    # map [-1,1] -> [0,1]
    if no_pp == False:
        true = np.clip((true), 0, 1)
        pred = np.clip((pred), 0, 1)
        rx = np.clip((rx), -0.5, 0.5) + 0.5
    
    canvas = np.zeros((3*image_shape[0], num_image*image_shape[1],3))
    for i in range(num_image):
        canvas[0:image_shape[0], image_shape[1]*i:image_shape[1]*i+image_shape[1]] = true[i,:,:,:]
        canvas[1*image_shape[0]:2*image_shape[0], image_shape[1]*i:image_shape[1]*i+image_shape[1]] = rx[i,:,:,:] 
        canvas[2*image_shape[0]:3*image_shape[0], image_shape[1]*i:image_shape[1]*i+image_shape[1]] = pred[i,:,:,:] 
    if is_show:
        plt.ion()
        plt.figure(0)
        plt.imshow(canvas, interpolation='nearest', vmin=0, vmax=1)
        plt.show()
        plt.pause(3)
    else:
        plt.imsave(prefix + "_Iter_"+str(iter)+postfix+".png", canvas, vmin=0, vmax=1)

def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(input=x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], ds_x[4]])
    
    y = tf.compat.v1.depth_to_space(input=x, block_size=block_size)
    
    ds_y = tf.shape(input=y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x
    
def DownSample3DMatlab(x, scale, method='cubic', antialiasing=True):
    ds_x = tf.shape(input=x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], 3])
    
    y = DownSample2DMatlab(x, scale, method, antialiasing)

    ds_y = tf.shape(input=y)
    y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return y
    
def DynFilter3D(x, F, filter_size):
    '''
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    '''
    
    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    filter_localexpand = tf.Variable(filter_localexpand_np, trainable=False, dtype='float32',name='filter_localexpand') # 5,5,5,125        
    x = tf.transpose(a=x, perm=[0,2,3,1]) # b, h, w, t
    x_localexpand = tf.nn.conv2d(input=x, filters=filter_localexpand, strides=[1,1,1,1], padding='SAME') # batch, 32,32,125
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, 32,32,1,125
    x = tf.matmul(x_localexpand, F) # b, 32,32,1,16
    x = tf.squeeze(x, axis=3) # b, 32,32,16

    return x
    
    
def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

    
h = gkern(kernlen=13, nsig=1.6)
h = h[:,:,np.newaxis,np.newaxis].astype(np.float32)

def DownSample(x, h, scale=4):
    ds_x = tf.shape(input=x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], 3])
    
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
    pad_array = [[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]]
    
    depthwise_F = tf.tile(W, [1, 1, 3, 1])
    y = tf.nn.depthwise_conv2d(input=tf.pad(tensor=x, paddings=pad_array, mode='REFLECT'), filter=depthwise_F, strides=[1, scale, scale, 1], padding='VALID')
    
    ds_y = tf.shape(input=y)
    y = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], 3])
    return y



stp = [[0,0], [1,1], [1, 1], [1, 1], [0,0]]
sp = [[0,0], [0,0], [1, 1], [1, 1], [0,0]]
sp5 = [[0,0], [0,0], [2, 2], [2, 2], [0,0]]

    
def G(x): # generate DUF, residual
    # x : b,5,32,32,3
    Fx, Rx = FR(x) #Fx: b,2,32,32,75,16
    Rx = depth_to_space_3D(Rx, 4)
    x_c = []
    x_f = []
    for f in range(1):
        for c in range(3):
            t = DynFilter3D(x[:,f+3:f+4,:,:,c], Fx[:,f,:,:,:,:], [1,5,5])  #b,32,32,16
            t = tf.compat.v1.depth_to_space(input=t, block_size=4) #b,128,128,1
            t = tf.squeeze(t, axis=3)  #b,128,128
            x_c += [t]
        x_f += [tf.stack(x_c, axis=3)] #b,128,128,3
        x_c = []
    x = tf.stack(x_f, axis=1) #b,2,128,128,3
    x += Rx
    
    return x, tf.reshape(Fx[:,0,16,16,:,0], [-1, 1, 5, 5, 1]), Rx
    
   
#with tf.device('gpu:1'):
H = tf.compat.v1.placeholder(tf.float32, shape=[None, 7, None, None, 3])
L_ = DownSample(H, h, 4)
L = L_[:,:,2:-2,2:-2,:]

is_train = tf.compat.v1.placeholder(tf.bool, shape=[]) # Phase ,scalar
lr_G = tf.compat.v1.placeholder(tf.float32, shape=[]) 

with tf.compat.v1.variable_scope('G') as scope:
    GL, Fx, Rx = G(L)

loss_M = Huber(H[:,3,24:-24,24:-24,:], GL[:,0,16:-16,16:-16,:], 0.01)
loss_M *= 1.

# Total loss
loss_G = loss_M 


trainable_G = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('G/')]
for v in trainable_G:
    print(v)

#opt = tf.train.AdamOptimizer(lr_G)
#grads_and_vars = opt.compute_gradients(loss_G, var_list=trainable_G)
#for g,v in grads_and_vars:
#    print(g)
#    print(v)
#grads_and_vars = [(tf.clip_by_norm(x[0], 0.1), x[1]) for x in grads_and_vars]
#train_G = opt.apply_gradients(grads_and_vars)
train_G = tf.compat.v1.train.AdamOptimizer(lr_G).minimize(loss_G, var_list=trainable_G)

params_G = [v for v in tf.compat.v1.global_variables() if v.name.startswith('G/')]

    
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = '1'

with tf.compat.v1.Session(config=config) as sess:
    tf.compat.v1.global_variables_initializer().run()
    resume = 0
    start_flag = True

    # restore v13
#    LoadParams(sess, [params_G], in_file='checkpoints/DF/v82_i190000.h5')

    
    curr_lr_G = 0.001
    best_vid4 = 25*4
    dT = 0.
    rT = 0.
    for i in range(resume, 220001):
        if i % 1000 == 0 and i != 0:
            
            # VSRval4
            valScenes = ['coastguard', 'foreman', 'garden', 'husky']
            n = 0
            vid4 = 0
            val_Gs = []
            for valScene in valScenes:
                fnames = glob.glob('./Val_mat/VSRVal4/'+valScene+'/*.png')
                fnames.sort()
                
                val_Hs = []
                for fname in fnames:                
                    val_Hs.append(_load_img_array(fname))
    
                val_H = np.asarray(val_Hs)
                val_G = np.empty_like(val_H)
                
                val_H_ = np.lib.pad(val_H, pad_width=((3,3),(0,0),(0,0),(0,0)), mode = 'constant')
                val_H_ = np.lib.pad(val_H_, pad_width=((0,0),(8,8),(8,8),(0,0)), mode = 'reflect')
                
                for f in range(0,val_H_.shape[0]-6):
                    in_H = val_H_[f:f+7] #select 4 frames
                    in_H = in_H[np.newaxis,:,:,:,:]
                    
                    out_G = sess.run(GL, feed_dict={H: in_H, is_train: False})
                    out_G = np.clip(out_G[0,0], 0. , 1.) 
#                    plt.imsave('validation/'+MODEL+'/v'+str(VERSION)+'_i'+ str(i) + '_Vid'+str(n)+'_f'+str(f)+".png", (out_G+1.)/2., vmin=0, vmax=1)
                    val_G[f] = out_G
                    
                _psnr = AVG_PSNR(((val_H)*255).astype(np.uint8)/255.0, ((val_G)*255).astype(np.uint8)/255.0, vmin=0, vmax=1, t_border=1, sp_border=16)
    
                n += 1
                vid4 += _psnr
                val_Gs += [val_G]  
                print('#{}: {}'.format(n, _psnr))
            print('Val4: {}'.format(vid4/4.))
            
            if vid4 > best_vid4:
                print('Saving the Best')
                n = 0
                best_vid4 = vid4
                for vid in val_Gs:
                    f = 0
                    for frame in vid:
                        plt.imsave('validation/'+MODEL+'/v'+str(VERSION)+'_i'+ str(i) + '_Vid'+str(n)+'_f'+str(f)+".png", (frame), vmin=0, vmax=1)
                        f += 1
                    n += 1
                SaveParams(sess, [params_G], out_file='checkpoints/'+MODEL+'/Best_v'+str(VERSION)+'_i{:d}.h5'.format(i))    

        t = time.time()
        batch_H = Iter_H.dequeue()
        dT += time.time()-t

        t = time.time()
        l_M, l_G, _ = sess.run([loss_M, loss_G, train_G], feed_dict={H: batch_H, is_train: True, lr_G:curr_lr_G})
        rT += time.time()-t
        
        if i% 100 == 0:
            print('I : {:4d} | M: {: 4.3e} | G: {: 4.3e} | dT: {: 4.3f} | rT: {: 4.3f} s'.format(i, l_M, l_G, dT/100., rT/100.))
            dT = 0.
            rT = 0.
            
            batch_G, batch_R = sess.run([GL, Rx], feed_dict={H: batch_H, is_train: False})
            plot_images(batch_H[:8,1,24:-24,24:-24], batch_R[:8,0,16:-16,16:-16], batch_G[:8,0,16:-16,16:-16], 4, [96,96], iter=i, is_show=False, prefix='validation/DF_show/v'+str(VERSION), postfix='_f0')

        if (i% 10000 == 0) and i != 0:
            SaveParams(sess, [params_G], out_file='checkpoints/'+MODEL+'/v'+str(VERSION)+'_i{:d}.h5'.format(i))
            
        if i%100000 == 0 and i != 0:
            curr_lr_G = np.maximum(0.1*curr_lr_G, 0.00001)
            print('lr:', curr_lr_G)
            
    print('Done')
