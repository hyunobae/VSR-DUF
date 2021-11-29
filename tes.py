import tensorflow as tf
import numpy as np
import tensorflow.keras
from nets import FR_52L
from utils import *

x_train = load_datasets("dataset/G")

mf, mr = FR_52L(x_train)