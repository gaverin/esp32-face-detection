import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

