# Based on 
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting
# https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

from tensorflow import keras
import pandas as pd
import numpy as np
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
        # DFs
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.X_train_normalized = None
        self.X_test_normalized = None
        
        #NP arrays
        self.X_train_unrolled = None
        self.X_test_unrolled = None
        
        self.model = None
        self.scaler = None
        
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
            
            
        
        
    # ===============================  Normalize, Power Transform, Split to Sqw =================================
    def normalize_X(self, scaler=None):
        """
        normalizes data with scaler (default is StandardScaler)
        """
        
        self.scaler = StandardScaler() if scaler is None else scaler()
        
        self.X_train_normalized = self.scaler.fit_transform(self.X_train)
        self.X_test_normalized = self.scaler.transform(self.X_test)
                
        return True
    
    # ------------------------------------------------------------------------------------------------------------------#    
    @staticmethod
    def _unroll_array_to_sequence(X, y, sequence_length=4):
        # src: https://analyticsindiamag.com/anomaly-detection-in-temperature-sensor-data-using-lstm-rnn-model/
        
        if len(X) != len(y):
            print(">>> X and y data should have same len...")
            return False
        
        set_X = []
        set_y = []
        
        for index in range(len(X) - sequence_length):
            set_X.append(X[index: index + sequence_length])
            set_y.append(y[index: index + sequence_length])
        
        return np.asarray(set_X), np.asarray(set_y)

    # ------------------------------------------------------------------------------------------------------------------#
    def split_X_to_sequencees(self, sequence_len):
        """
        Splits X to sequences
        >>> Example: [1,2,3,4,5] n_steps/ sequence_len=3 --> [1,2,3], [2,3,4], [3,4,5]
        """
        self.X_train_unrolled, self.y_train_unrolled = self._unroll_array_to_sequence(
            self.X_train_normalized, 
            self.y_train.values, 
            sequence_len)
        
        self.X_test_unrolled, self.y_test_unrolled = self._unroll_array_to_sequence(
            self.X_test_normalized, 
            self.y_test.values, 
            sequence_len)    
        # ------------------------------------------------------------------------------------------------------------------#
    # TODO: PowerTransform

    # ========================================  Model Creation  ==============================================
    def model_combine(self, template:list, compile_model=True, compile_dict=None, verbose=True):
        """
        Combines self.model from template
        
        :param template: list with layers
            >>> example: 
                template = [
                            TimeDistributed(Conv1D(**conv1D_params, input_shape=(None, n_steps, n_features))),
                            TimeDistributed(MaxPooling1D(pool_size=2)),
                            TimeDistributed(Flatten()),
                            LSTM(50, activation='relu'),
                            Dense(1)
                        ]
        :param compile_model: if True, compile_dict has to be provided
        """
        
        self.model = keras.Sequential()
        for layer in template:                
            self.model.add(layer)
        
        if compile_model and (compile_dict is None):
            compile_dict = dict()
            compile_defaults = dict(optimizer='adam', loss='mse', metrics=['mae'])            
            self.helpers_set_dict_default(compile_dict, compile_defaults)
        
        self.model.compile(**compile_dict)
        print(">>> model compiled")
        
        if verbose:
            self.model.summary()
        
        return True
    
    # ------------------------------------------------------------------------------------------------------------------#
    def model_fit(epoches=None, print_charts=True, return_results=False):
        """
        Fits the self.model
        """
        if self.model is None:
            print(">>>No model detected. First you have to combine it...")
            return False
        
#         results = self.model.fit()
        

    # ========================================  Helpers  ==============================================
    @staticmethod
    def helpers_set_dict_default(dictionar, keys):
        """
        Sets values for keys in dict.
        
        :param dictionar: dictionary that has to be inspected for presence of keys
        :param keys:
            >> list: keys from list that are not found in dictionar, will be added with None value
            >> dict: keys from keys dict that are not found in dictionar, will be added with keys[key] value
                
        
        returns: None --> as dict is passed by ref, no real need for return
        
        """
        if isinstance(keys, list):
            for key in keys:
                dictionar[key] = dictionar[key] if key in dictionar else None
        
        if isinstance(keys, dict):
            for key, val in keys.items():
                dictionar[key] = dictionar[key] if key in dictionar else val