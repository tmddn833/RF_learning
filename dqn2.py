"""
this source code is modified from dqn with tf 1 the old version.
try to change the version of tensorflow and make it brand new!
"""

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

class DQN:
    ''' there is no session argument in tf2 '''
    def __init__(self, input_size: int, output_size: int, name: str="main") -> None:
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._build_network()

    def _build_network(self, h_size=32, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """

        self._X = Input(shape=self.input_size, name="input_x")
        net = Dense(units= h_size, activation=tf.nn.relu,kernel_initializer='glorot_uniform')(self._X)
        # glorot_uniform is  Xavier normal initializer
        output = Dense(units=self.output_size)(net)
        self._Qpred = Model(inputs=[self._X], outputs=[output])

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        x = np.reshape(state, [-1, self.input_size])
        # input_tensor = tf.convert_to_tensor(x)
        # output_tensor = self._Qpred(input_tensor)
        # output_array = output_tensor.numpy()
        # return output_array
        prediction = self._Qpred(x)

        return prediction.numpy()

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        self._Qpred.compile(loss='mse', optimizer='Adam')
        history = self._Qpred.fit(x=x_stack, y=y_stack, verbose=0)
        #print(history.history)
        return history.history['loss']