import numpy as np
import sys
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

#############################################################################
###########################    TRAIN MODEL     ##############################
#############################################################################

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, maxMemorySize, minMemorySize):
        self._inputDim = input_dim
        self._outputDim = output_dim
        self._batchSize = batch_size
        self._learningRate = learning_rate
        self._model = self.build_model(num_layers, width)
        # Memory
        self._samples = []
        self._maxMemorySize = maxMemorySize
        self._minMemorySize = minMemorySize


    def build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._inputDim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._outputDim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        import tensorflow as tf
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=self._learningRate))
        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._inputDim])
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the NN using the updated Q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)

    @property
    def input_dim(self):
        return self._inputDim

    @property
    def output_dim(self):
        return self._outputDim

    @property
    def batch_size(self):
        return self._batchSize


    # Memory
    def add_sample(self, sample):
        """
        Adding a sample into the memory
        """
        self._samples.append(sample)
        if self.size_now() > self._maxMemorySize:
            self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element

    def get_samples(self, n):
        """
        Get n samples randomly from the memory
        """
        if self.size_now() < self._minMemorySize:
            return []

        if n > self.size_now():
            return random.sample(self._samples, self.size_now())  # get all the samples
        else:
            return random.sample(self._samples, n)  # get "batch size" number of samples

    def size_now(self):
        """
        Check how full the memory is
        """
        return len(self._samples)

#############################################################################
###########################    TEST MODEL     ###############################
#############################################################################

class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self.LoadMyModel(model_path)


    def LoadMyModel(self, modelFolderPath):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        ModelFilePath = os.path.join(modelFolderPath, 'trained_model.h5')
        
        if os.path.isfile(ModelFilePath):
            LoadedModel = load_model(ModelFilePath)
            return LoadedModel
        else:
            sys.exit("Could not found the given MODEL NUMBER !!")


    def predict_one(self, state):
        """
        This function return the predicted action values from the given state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    @property
    def input_dim(self):
        return self._input_dim

