# Based on 
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting
# https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

from tensorflow import keras
import pandas as pd
import glob, os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class NeuralManager:
    """
    SR: operates over NN:
    inits data. One instance has to be created for one set of data (cut, long, full...)
    combines the NN from user pattern, trains, saves the best weights
    """
    # ===============================  Init  ====================================================
    def __init__(self, dir_path=None):
        """
        grabs all possible train_test splits from the dir and attaches to self.
        """
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.model = keras.Sequential()
        
        self._train_test_init = dict()
        self._init_train_test_data(dir_path)
        
    
    def _init_train_test_data(self, dir_path, verbose=True):
        """
        gets train test data
        
        :param dir_path: dir where train test files located
        """
        for file_path in [f for f in glob.glob(os.path.join(dir_path, "*.csv"))]:
            file_name =  file_path.split('/')[-1].split('.')[0]
            
            if "X_train" in file_name:
                self.X_train = pd.read_csv(file_path, index_col="Date")
                self._train_test_init[file_name] = True
            if "X_test" in file_name:
                self.X_test = pd.read_csv(file_path, index_col="Date")
                self._train_test_init[file_name] = True
            if "y_train" in file_name:
                self.y_train = pd.read_csv(file_path, index_col="Date")
                self._train_test_init[file_name] = True
            if "y_test" in file_name:
                self.y_test = pd.read_csv(file_path, index_col="Date")
                self._train_test_init[file_name] = True
                
        if verbose:
            print(">>> train-test inited: " , self._train_test_init)
            
    # ===============================  Normalize, Power Transform  ==============================================
    @staticmethod
    def normalize_dataset(X_train=None, X_test=None, scaler=None):
        """
        Fits on Train set, transforms both Train and Test
        
        returns mnormalized 2D arrays
        """
        if scaler is None:
            scaler = StandardScaler
        return scaler.fit_transform(X_train), scaler.transform(X_test)
    # ------------------------------------------------------------------------------------------------------------------#
    # TODO: PowerTransform

    # ========================================  Model Creation  ==============================================
    def combine_model(self, template:list):
        """
        Combines model from template
        
        :param template: list with layers
            >>> example: [["Conv2D", dict(**kwargs)], ["MaxPooling2D", dict(**kwargs)]...]]
                kwargs: kernel_size, padding, stride...
                input_shape: should come with the first layer in template
        """
        
        for i, layer in enumerate(template):
            if 'input_shape' not in layer[1]:
                print("input_shape is a required param...")
                return
                
            self.model.add(layer)
        
        