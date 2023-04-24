import tensorflow as tf

class CFNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, num_units, **kwargs) -> None:
        super(CFNCell,self).__init__(**kwargs)
        self._num_units = num_units
        # self.state_size = tf.TensorShape([num_units])
        # self.output_size = tf.TensorShape([num_units])
        self._activation = tf.keras.activations.tanh


    @property
    def state_size(self):
      return self._num_units

    def build(self, inputs_shape):
        #super().build(inputs_shape)
        self._weights_U_theta = self.add_weight(
            "gate/U_theta",
            shape=(self._num_units, self._num_units),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.07,maxval=0.07),
            trainable=True   #
            )
    
        self._weights_V_theta = self.add_weight(
            "gate/V_theta",
            shape=(inputs_shape[-1], self._num_units),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.07,maxval=0.07),   ##tf.random_uniform_initializer(dtype = self.dtype, minval=-0.07,maxval=0.07)
            trainable=True
            )
        
        self._bias_theta = self.add_weight(
            "gate/b_theta",
            shape=(self._num_units),
            initializer= tf.keras.initializers.Constant(value=1), ##tf.ones_initializer(dtype=self.dtype))
            trainable=True
            )
        
        self._weights_U_n = self.add_weight(
            "gate/U_n",
            shape=(self._num_units, self._num_units),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.07,maxval=0.07),
            trainable=True
            )
        
        self._weights_V_n = self.add_weight(
            "gate/V_n",
            shape=(inputs_shape[-1], self._num_units),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.07,maxval=0.07),
            trainable=True
            )
        
        self._bias_n = self.add_weight(
            "gate/b_n",
            shape=(self._num_units),
            initializer= tf.keras.initializers.Constant(value=-1), ##MinusOnes(dtype=self.dtype)
            trainable=True
            )
        self._W =  self.add_weight(
            "kernel",
            shape=(inputs_shape[-1], self._num_units),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.07,maxval=0.07),
            trainable=True
            )

        self.built = True

    def call(self, inputs, state):
        states = state[0]

        theta = tf.keras.activations.sigmoid(
            tf.add(tf.add(tf.matmul(states, self._weights_U_theta),
                    tf.matmul(inputs, self._weights_V_theta)),
                    self._bias_theta)
        )
        eta = tf.keras.activations.sigmoid(
            tf.add(tf.add(tf.matmul(states, self._weights_U_n),
                    tf.matmul(inputs, self._weights_V_n)),
                    self._bias_n)
        )
        output = tf.add(
                    tf.math.multiply(theta, self._activation(states)),
                    tf.math.multiply(eta, self._activation(tf.matmul(inputs, self._W)))
        )
        return output ,output
