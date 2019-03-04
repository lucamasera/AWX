import numpy as np
import keras.backend as K
import tensorflow as tf
import networkx as nx

from keras.layers import Dense


class AWX(k.layers.Dense):
    def __init__(
        self, A, n_norm=5, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs):
        
        assert n_norm >= 0
        self.use_soft_clipping = use_soft_clipping
        self.max_val_clip = max_val_clip
        self.clip_base = clip_base
        
        self.n = n_norm
        self.leaves = A.sum(0) == 0
        units = sum(self.leaves)
        self.A = deepcopy(A)
        
        R = np.zeros(A.shape)
        R[self.leaves,self.leaves] = 1
        
        self.g = nx.DiGraph(A)
        
        for i in np.where(self.leaves)[0]:
            ancestors = list(nx.descendants(self.g, i))
            if ancestors:
                R[i, ancestors] = 1
        
        self.R = K.constant(R[self.leaves])
        self.R_t = K.constant(R[self.leaves].T)
        
        super(AWX, self).__init__(
            units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, 
            **kwargs
        )
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.R.shape[1]
        return tuple(output_shape)
    
    def n_norm(self, x, epsilon=1e-6):
        return K.pow(K.clip(K.sum(K.pow(x, self.n), -1), epsilon, 1-epsilon), 1./self.n)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
            
        if self.activation is not None:
            output = self.activation(output)
        
        if self.n > 1:
            output = self.n_norm(tf.multiply(tf.expand_dims(output, 1), self.R_t))
        elif self.n > 0:
            output = K.minimum(K.constant(1-1e-4), K.sum(tf.multiply(tf.expand_dims(output, 1), self.R_t), -1))
        else:
            output = K.max(tf.multiply(tf.expand_dims(output, 1), self.R_t), -1)
            
        return output
