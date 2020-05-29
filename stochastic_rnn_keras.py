from keras.layers import LSTM
from keras import initializers
from tensorflow.keras import backend as K


def get_mask(batch_size, dim, p):
    """Sample bernoulli mask from concrete distribution
    p: dropout rate"""
    t = 1e-1
    eps = K.epsilon()

    u = K.random_uniform(shape=(4, batch_size, dim))
    z = (1-K.sigmoid((K.log(p+eps) - K.log(1-p+eps) + K.log(u+eps) - K.log(1-u+eps)) / t))/(1-p)

    return z


class StochasticLSTM(LSTM):
    """StochasticLSTM that apply dropout to input and hidden state
    Note 1: do not set regularizers because dropout regularizers will be applied
    Note 2: there are 2 dropout rates: dropout for input, and recurrent_dropout for hidden state
    Note 3: to enable learning dropout rates, set dropout rates to 1.0"""

    def build(self, input_shape):
        super().build(input_shape)
        
        reg = 1/14681
        dropout_reg = 2/14681
        def dropout_constraint(p):
            """Constraint probability between 0.0 and 1.0"""
            return K.clip(p, K.epsilon(), 1. - K.epsilon())

        if self.dropout == 1.0:
            self.p = self.cell.add_weight(name='p',
                                          shape=(),
                                          initializer=initializers.uniform(minval=0.3, maxval=0.7),
                                          constraint=dropout_constraint,
                                          trainable=True)
            self.add_loss(dropout_reg*input_shape[-1] *
                          (self.p * K.log(self.p) +
                           (1-self.p) * K.log(1-self.p)))
        else:
            self.p = self.dropout

        if self.recurrent_dropout == 1.0:
            self.p_r = self.cell.add_weight(name='p_recurrent',
                                          shape=(),
                                          initializer=initializers.uniform(minval=0.3, maxval=0.7),
                                          constraint=dropout_constraint,
                                          trainable=True)
            self.add_loss(dropout_reg*self.units *
                          (self.p_r * K.log(self.p_r) +
                           (1-self.p_r) * K.log(1-self.p_r)))
        else:
            self.p_r = self.recurrent_dropout

        # weight loss
        self.add_loss(reg / (1.-self.p) * K.sum(K.square(self.cell.kernel)))
        self.add_loss(reg / (1.-self.p_r) * K.sum(K.square(self.cell.recurrent_kernel)))
        self.add_loss(reg * K.sum(K.square(self.cell.bias)))
        
        self.built = True
    
    def call(self, inputs, mask=None, training=None, initial_state=None):
        input_shape = K.shape(inputs)
        B = input_shape[0]
        D = input_shape[2]
        self.cell._dropout_mask = get_mask(B, D, self.p)
        self.cell._recurrent_dropout_mask = get_mask(B, self.units, self.p_r)
        return super(LSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)
