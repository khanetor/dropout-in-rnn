from tensorflow.keras.layers import concatenate, Layer
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K

import numpy as np

class StochasticLSTM(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        super(StochasticLSTM, self).__init__(**kwargs)
        
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.go_backwards = go_backwards
        self.unroll = unroll
        
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
    @property
    def dropout_rate(self):
        if isinstance(self.dropout, float):
            return self.dropout
        else:
            return self.dropout[0].numpy()

    def build(self, input_shape):
        D = input_shape[-1]
        units = self.units
        ls = 1e-2
        tau = 1.

        if self.dropout is None:
            def dropout_regularizer(p):
                return D * (p * K.log(p) + (1-p) * K.log(1-p))
            
            def dropout_constraint(p):
                return K.clip(p, K.epsilon, 1. - K.epsilon())

            self.dropout = self.add_weight(name='dropout',
                                           shape=(1,),
                                           initializer=initializers.RandomUniform(minval=0.2, maxval=.8),
                                           regularizer=dropout_regularizer,
                                           constraint=dropout_constraint,
                                           trainable=True)

        self.kernel_regularizer = regularizers.l2(0.5*ls**2 * (1.-self.dropout) / tau)
        self.recurrent_regularizer = regularizers.l2(0.5*ls**2 * (1.-self.dropout) / tau)
        self.bias_regularizer = regularizers.l2(0.5*ls**2 / tau)
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(D, self.units*4),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        
        self.recurrent_kernel = self.add_weight(name='recurrent_kernel',
                                        shape=(self.units, self.units*4),
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint,
                                        trainable=True)
        #Segments weights
        self.kernel_i = self.kernel[:,         : units  ]
        self.kernel_f = self.kernel[:, units   : units*2]
        self.kernel_o = self.kernel[:, units*2 : units*3]
        self.kernel_g = self.kernel[:, units*3 :        ]

        self.recurrent_kernel_i = self.recurrent_kernel[:,         : units  ]
        self.recurrent_kernel_f = self.recurrent_kernel[:, units   : units*2]
        self.recurrent_kernel_o = self.recurrent_kernel[:, units*2 : units*3]
        self.recurrent_kernel_g = self.recurrent_kernel[:, units*3 :        ]
        
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(name='bias',
                                        shape=(self.units*4,),
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
            self.bias_i = self.bias[        : units  ]
            self.bias_f = self.bias[units   : units*2]
            self.bias_o = self.bias[units*2 : units*3]
            self.bias_g = self.bias[units*3 :        ]
        
        super(StochasticLSTM, self).build(input_shape)
        
    def _get_mask(self, batch_size, dim):
        B, D, H = batch_size, dim, self.units
        t = 1e-1
        eps = K.epsilon()
        
        if B is None:
            return K.placeholder(shape=(4, B, D)), K.placeholder(shape=(4, B, H))
        
        if dim == 1:
            px = 0.0
        else:
            px = self.dropout
        ph = self.dropout
        
        ux = K.random_uniform(shape=(4, B, D))
        uh = K.random_uniform(shape=(4, B, H))

        zx = (1-K.sigmoid((K.log(px+eps) - K.log(1-px+eps) + K.log(ux+eps) - K.log(1-ux+eps)) / t))/(1-px)
        zh = (1-K.sigmoid((K.log(ph+eps) - K.log(1-ph+eps) + K.log(uh+eps) - K.log(1-uh+eps)) / t))/(1-ph)
        return zx, zh
    
    def _get_initial_states(self, batch_size):
        B, units = batch_size, self.units
        if B is None:
            return [K.placeholder(shape=(B, units)), K.placeholder(shape=(B, units))]
        else:
            return [K.zeros((B, units)), K.zeros((B, units))]
        
    def call(self, inputs, mask=None, training=None, initial_state=None):
        input_shape = K.int_shape(inputs)
        if len(input_shape) != 3:
            raise ValueError("Input 0 of layer %s is incompatible with the layer: expected ndim=3, found ndim=%d. Full shape received: %s" % (self.name, len(input_shape), input_shape))
        B, T, D = input_shape
        units = self.units

        zx, zh = self._get_mask(B, D)

        if initial_state is None:
            initial_states = self._get_initial_states(B)
        else:
            initial_states = initial_state

        def _step_fn(x, states):
            h, c = states
            units = self.units

            x_i, x_f, x_o, x_g = x*zx[0], x*zx[1], x*zx[2], x*zx[3]
            h_i, h_f, h_o, h_g = h*zh[0], h*zh[1], h*zh[2], h*zh[3]

            M_i = K.dot(h_i, self.recurrent_kernel_i) + K.dot(x_i, self.kernel_i)
            M_f = K.dot(h_f, self.recurrent_kernel_f) + K.dot(x_f, self.kernel_f)
            M_o = K.dot(h_o, self.recurrent_kernel_o) + K.dot(x_o, self.kernel_o)
            M_g = K.dot(h_g, self.recurrent_kernel_g) + K.dot(x_g, self.kernel_g)
            
            if self.use_bias:
                M_i = K.bias_add(M_i, self.bias_i)
                M_f = K.bias_add(M_f, self.bias_f)
                M_o = K.bias_add(M_o, self.bias_o)
                M_g = K.bias_add(M_g, self.bias_g)

            i = self.recurrent_activation(M_i)
            f = self.recurrent_activation(M_f)
            o = self.recurrent_activation(M_o)
            g = self.activation(M_g)

            new_c = f * c + i * g
            new_h = o * self.activation(new_c)

            return new_h, [new_h, new_c]

        last_output, outputs, states = K.rnn(_step_fn, inputs, initial_states,
                                             go_backwards=self.go_backwards, mask=mask, unroll=self.unroll)
        if self.return_sequences:
            output = outputs
        else:
            output = last_output
            
        if self.return_state:
            return [output] + list(states)
        else:
            return output
    
    def compute_output_shape(self, input_shape):
        B, T, D = input_shape
        units = self.units
        
        if self.return_sequences:
            output_shape = (B, T, units)
        else:
            output_shape = (B, units)
        
        if self.return_state:
            return [output_shape, (B, units), (B, units)]
        else:
            return output_shape


def mc_sample(model, inputs, T:int=10):
    output_shape = model.compute_output_shape(inputs.shape)
    result = np.empty(shape=(T, *output_shape))
    for t in range(T):
        result[t] = model(inputs)
    return result
