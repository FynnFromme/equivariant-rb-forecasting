from tensorflow import keras
import tensorflow as tf
from networks.rayleigh_benard import se2n_cnn

L2 = 0

def build(horizontal_size, height, rb_channels, batch_size, G='D4', v_sharing=1):
    model = keras.Sequential([
            keras.layers.InputLayer(shape=(horizontal_size, horizontal_size, height, rb_channels),
                                    batch_size=batch_size),
            
            ##########################
            #    Data Augmentation   #
            ##########################
            keras.layers.Reshape((horizontal_size, horizontal_size, height*rb_channels)),
            keras.layers.RandomRotation(factor=1, fill_mode='wrap', interpolation='bilinear'),
            keras.layers.RandomFlip(mode='horizontal_and_vertical'),
            keras.layers.Reshape((horizontal_size, horizontal_size, height, rb_channels)),
            
            ###############
            #   Encoder   #
            ###############
            se2n_cnn.RB3D_LiftDN_Conv(8, v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=4, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            se2n_cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            keras.layers.Dropout(rate=0.2),
            se2n_cnn.RB3D_DN_Conv(v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=6, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            se2n_cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            keras.layers.Dropout(rate=0.2),
            se2n_cnn.RB3D_DN_Conv(v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=8, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            se2n_cnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            keras.layers.Dropout(rate=0.2),
            se2n_cnn.RB3D_DN_Conv(v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=12, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            # gcnn.SpatialPooling(ksize=(2,2,2), pooling_type='MAX', strides=(2,2,2), padding='VALID'),
            
            ###############
            #   Decoder   #
            ###############
            # gcnn.UpSampling(size=(2,2,2)),
            keras.layers.Dropout(rate=0.2),
            se2n_cnn.RB3D_DN_Conv(v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=12, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            se2n_cnn.UpSampling(size=(2,2,2)),
            keras.layers.Dropout(rate=0.2),
            se2n_cnn.RB3D_DN_Conv(v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=8, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            se2n_cnn.UpSampling(size=(2,2,2)),
            keras.layers.Dropout(rate=0.2),
            se2n_cnn.RB3D_DN_Conv(v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=6, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.BatchNorm(),
            keras.layers.Activation('relu'),
            
            se2n_cnn.UpSampling(size=(2,2,2)),
            keras.layers.Dropout(rate=0.2),
            se2n_cnn.RB3D_DN_Conv(v_sharing=v_sharing, h_ksize=3, v_ksize=5, channels=rb_channels, h_padding='WRAP', v_padding='SAME', 
                             strides=(1,1,1), filter_initializer='he_normal', use_bias=False,
                             filter_regularizer=keras.regularizers.L2(L2)),
            se2n_cnn.TransformationPooling(tf.reduce_mean, keepdims=False)
            ])

    return model