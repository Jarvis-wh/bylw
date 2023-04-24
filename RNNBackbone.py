from keras.layers import RNN
import tensorflow as tf
from keras import backend
from CFN_impl import CFNCell
batch_size, num_steps = 32, 35


# First, let's define a RNN Cell, as a layer subclass.
from CFN_impl import CFNCell
# Let's use this cell in a RNN layer:

cell = CFNCell(128,0.1,0.1)
rnn_layer = tf.keras.layers.RNN(cell, time_major=True,
    return_sequences=True, return_state=True)
state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # rnn返回两个以上的值
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)
    