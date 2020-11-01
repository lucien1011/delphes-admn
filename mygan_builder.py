import numpy

import tensorflow as tf
import tensorflow.keras.backend as K

def build_model(cfg):
    x_input_layer = tf.keras.layers.Input(shape=(cfg.ndf,),name="x_input",)
    condition_input_layer = tf.keras.layers.Input(shape=cfg.condition_input_shape,name="condition_input",)

    discriminator = Discriminator(
            cfg.disc_neuron_list,
            )
    generator = Generator(
            cfg.ndf,
            cfg.gen_neuron_list,
            )
    return discriminator,generator,x_input_layer,condition_input_layer

class Discriminator(tf.keras.Model):
    def __init__(self,
            neuron_number_list,
            **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.neuron_number_list = neuron_number_list
        self._dense_layer_list = []
        self._dropout_layer_list = []
        self._leaky_layer_list = []
        self._batchNorm_layer_list = []
        for i,neuron_number in enumerate(neuron_number_list):
            self._dense_layer_list.append(tf.keras.layers.Dense(neuron_number,activation='relu',))
            self._leaky_layer_list.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self._dropout_layer_list.append(tf.keras.layers.Dropout(0.1))
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, Training=True):
        x,condition = inputs
        x = tf.cast(x,tf.float32)
        condition = tf.cast(condition,tf.float32)
        disc_out = tf.concat([x,condition,], axis=-1)
        for i,neuron_number in enumerate(self.neuron_number_list):
            disc_out = self._dense_layer_list[i](disc_out)
            disc_out = self._leaky_layer_list[i](disc_out)
            disc_out = self._dropout_layer_list[i](disc_out)
            disc_out = tf.concat([disc_out,condition,], axis=-1)
        disc_out = self.output_layer(disc_out)
        return disc_out

class NormalSampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, 1))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class LogNormalSampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, 1))
        return z_mean * (1. + tf.exp(0.5 * z_log_var) * epsilon)

class Generator(tf.keras.Model):
    def __init__(self,
        ndf,
        neuron_number_list,
        **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.neuron_number_list = neuron_number_list
        self._dense_layer_list = []
        self._dropout_layer_list = []
        self._leaky_layer_list = []
        
        for i,neuron_number in enumerate(neuron_number_list):
            self._dense_layer_list.append(tf.keras.layers.Dense(neuron_number,activation='relu',))
            self._leaky_layer_list.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self._dropout_layer_list.append(tf.keras.layers.Dropout(0.1))
 
        self.recoPt_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="reco_pt_mean",)
        self.recoPt_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="reco_pt_sigma",)
        self.recoPt_sampling_layer = NormalSampling(name="reco_pt_sampling",)
        #self.recoPt_sampling_layer = LogNormalSampling(name="reco_pt_sampling",)
        
        self.recoEta_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="reco_eta_mean",)
        self.recoEta_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="reco_eta_sigma",)
        self.recoEta_sampling_layer = NormalSampling(name="reco_eta_sampling",)
        #self.recoEta_sampling_layer = LogNormalSampling(name="reco_eta_sampling",)
        
        self.recoPhi_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="reco_phi_mean")
        self.recoPhi_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="reco_phi_sigma")
        self.recoPhi_sampling_layer = NormalSampling(name="reco_phi_sampling",)
        #self.recoPhi_sampling_layer = LogNormalSampling(name="reco_phi_sampling",)

        self.trackDZ_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="trackDZ_mean",)
        self.trackDZ_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="trackDZ_sigma",)
        self.trackDZ_sampling_layer = NormalSampling(name="trackDZ_sample",)
        
        self.trackD0_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="trackD0_mean",)
        self.trackD0_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="trackD0_sigma",)
        self.trackD0_sampling_layer = NormalSampling(name="trackD0_sample",)

        self.T_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="T_mean",)
        self.T_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="T_sigma",)
        self.T_sampling_layer = NormalSampling(name="T_sample",)

        self.trackOuterx_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="trackOuterx_mean",)
        self.trackOuterx_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="trackOuterx_sigma",)
        self.trackOuterx_sampling_layer = NormalSampling(name="trackOuterx_sample",)

        self.trackOutery_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="trackOutery_mean",)
        self.trackOutery_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="trackOutery_sigma",)
        self.trackOutery_sampling_layer = NormalSampling(name="trackOutery_sample",)

        self.trackOuterz_mean_layer = tf.keras.layers.Dense(1,activation='linear',name="trackOuterz_mean",)
        self.trackOuterz_sigma_layer = tf.keras.layers.Dense(1,activation='linear',name="trackOuterz_sigma",)
        self.trackOuterz_sampling_layer = NormalSampling(name="trackOuterz_sample",)

    def call(self, condition, Training=True):
        gen_out = condition
        for i,neuron_number in enumerate(self.neuron_number_list):
            gen_out = self._dense_layer_list[i](gen_out)
            gen_out = self._leaky_layer_list[i](gen_out)
            gen_out = self._dropout_layer_list[i](gen_out)

        recoPt_sigma_out = self.recoPt_sigma_layer(gen_out)
        recoPt_out = self.recoPt_sampling_layer([
            tf.reshape(condition[:,0],(K.shape(condition)[0],1)),
            recoPt_sigma_out,
            ])

        recoEta_sigma_out = self.recoEta_sigma_layer(gen_out)
        recoEta_out = self.recoEta_sampling_layer([
            tf.reshape(condition[:,1],(K.shape(condition)[0],1)),
            recoEta_sigma_out,
            ])
        
        recoPhi_sigma_out = self.recoPhi_sigma_layer(gen_out)
        recoPhi_out = self.recoPhi_sampling_layer([
            tf.reshape(condition[:,2],(K.shape(condition)[0],1)),
            recoPhi_sigma_out,
            ])

        trackD0_input_out = gen_out
        trackD0_mean_out = self.trackD0_mean_layer(trackD0_input_out)
        trackD0_sigma_out = self.trackD0_sigma_layer(trackD0_input_out)
        trackD0_out = self.trackD0_sampling_layer([
            trackD0_mean_out,
            trackD0_sigma_out,
            ])

        trackDZ_input_out = gen_out
        trackDZ_mean_out = self.trackDZ_mean_layer(trackDZ_input_out)
        trackDZ_sigma_out = self.trackDZ_sigma_layer(trackDZ_input_out)
        trackDZ_out = self.trackDZ_sampling_layer([
            trackDZ_mean_out,
            trackDZ_sigma_out,
            ])
        
        T_mean_out = self.T_mean_layer(gen_out)
        T_sigma_out = self.T_sigma_layer(gen_out)
        T_normal_out = self.T_sampling_layer([
            T_mean_out,
            T_sigma_out,
            ])
        T_out = T_normal_out
        
        trackOuterx_input_out = gen_out
        trackOuterx_mean_out = self.trackOuterx_mean_layer(trackOuterx_input_out)
        trackOuterx_sigma_out = self.trackOuterx_sigma_layer(trackOuterx_input_out)
        trackOuterx_out = self.trackOuterx_sampling_layer([
            trackOuterx_mean_out,
            trackOuterx_sigma_out,
            ])

        trackOutery_input_out = gen_out
        trackOutery_mean_out = self.trackOutery_mean_layer(trackOutery_input_out)
        trackOutery_sigma_out = self.trackOutery_sigma_layer(trackOutery_input_out)
        trackOutery_out = self.trackOutery_sampling_layer([
            trackOutery_mean_out,
            trackOutery_sigma_out,
            ])
        
        trackOuterz_input_out = gen_out
        trackOuterz_mean_out = self.trackOuterz_mean_layer(trackOuterz_input_out)
        trackOuterz_sigma_out = self.trackOuterz_sigma_layer(trackOuterz_input_out)
        trackOuterz_out = self.trackOuterz_sampling_layer([
            trackOuterz_mean_out,
            trackOuterz_sigma_out,
            ])
        
        gen_out = tf.concat([
            recoPt_out,
            recoEta_out,
            recoPhi_out,
            T_out,
            trackD0_out,
            trackDZ_out,
            trackOuterx_out,
            trackOutery_out,
            trackOuterz_out,
            ],axis=-1)
        return gen_out

    def calculate_resolution(self, condition, Training=True):
        gen_out = condition
        for i,neuron_number in enumerate(self.neuron_number_list):
            gen_out = self._dense_layer_list[i](gen_out)
            gen_out = self._leaky_layer_list[i](gen_out)
            gen_out = self._dropout_layer_list[i](gen_out)

        recoPt_sigma_out = self.recoPt_sigma_layer(gen_out)
        recoEta_sigma_out = self.recoEta_sigma_layer(gen_out)
        recoPhi_sigma_out = self.recoPhi_sigma_layer(gen_out)

        trackD0_input_out = gen_out
        trackD0_mean_out = self.trackD0_mean_layer(trackD0_input_out)
        trackD0_sigma_out = self.trackD0_sigma_layer(trackD0_input_out)

        trackDZ_input_out = gen_out
        trackDZ_mean_out = self.trackDZ_mean_layer(trackDZ_input_out)
        trackDZ_sigma_out = self.trackDZ_sigma_layer(trackDZ_input_out)
        
        T_mean_out = self.T_mean_layer(gen_out)
        T_sigma_out = self.T_sigma_layer(gen_out)
        
        trackOuterx_input_out = gen_out
        trackOuterx_mean_out = self.trackOuterx_mean_layer(trackOuterx_input_out)
        trackOuterx_sigma_out = self.trackOuterx_sigma_layer(trackOuterx_input_out)

        trackOutery_input_out = gen_out
        trackOutery_mean_out = self.trackOutery_mean_layer(trackOutery_input_out)
        trackOutery_sigma_out = self.trackOutery_sigma_layer(trackOutery_input_out)
        
        trackOuterz_input_out = gen_out
        trackOuterz_mean_out = self.trackOuterz_mean_layer(trackOuterz_input_out)
        trackOuterz_sigma_out = self.trackOuterz_sigma_layer(trackOuterz_input_out)
        
        gen_out = tf.concat([
            #recoPt_sigma_out,
            #recoEta_sigma_out,
            #recoPhi_sigma_out,
            T_mean_out,
            T_sigma_out,
            trackD0_mean_out,
            trackD0_sigma_out,
            trackDZ_mean_out,
            trackDZ_sigma_out,
            trackOuterx_mean_out,
            trackOuterx_sigma_out,
            trackOutery_mean_out,
            trackOutery_sigma_out,
            trackOuterz_mean_out,
            trackOuterz_sigma_out,
            ],axis=-1)
        return gen_out.numpy()
