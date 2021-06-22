# Based on 
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting
# https://towardsdatascience.com/the-time-series-transformer-2a521a0efad3
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

from tensorflow import keras
import pandas as pd
import numpy as np
import glob, os, gc
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        self.y_pred = None # for predicted y
        
        self.X_train_normalized = None
        self.X_test_normalized = None
        
        #NP arrays
        self.X_train_unrolled = None
        self.X_test_unrolled = None
        
        self.model = None
        self.scaler = None
        
        # dict that stores data on which of the data collections were inited (X_train, X_test, y_train, y_test)
        self._train_test_init = dict() 
        self._init_train_test_data(dir_path)
        
        self.training_seq_params = dict()
        self.X_train_shape = list() # order = 'n_rows', 'seq_len', 'n_features'
        self.X_test_shape = list()
        
    
    def _init_train_test_data(self, dir_path, verbose=True):
        """
        gets train test data
        
        :param dir_path: dir where train test files located
        """
        for file_path in [f for f in glob.glob(os.path.join(dir_path, "*.csv"))]:
            file_name =  file_path.split('/')[-1].split('.')[0]
            
            if "X_train" in file_name:
                self.X_train = pd.read_csv(file_path, index_col="Date", parse_dates=True).sort_index(ascending=True)
                self._train_test_init['X_train'] = True
            if "X_test" in file_name:
                self.X_test = pd.read_csv(file_path, index_col="Date", parse_dates=True).sort_index(ascending=True)
                self._train_test_init['X_test'] = True
            if "y_train" in file_name:
                self.y_train = pd.read_csv(file_path, index_col="Date", parse_dates=True).sort_index(ascending=True)
                self._train_test_init['y_train'] = True
            if "y_test" in file_name:
                self.y_test = pd.read_csv(file_path, index_col="Date", parse_dates=True).sort_index(ascending=True)
                self._train_test_init['y_test'] = True
                
        if verbose:
            print(">>> train-test inited: " , self._train_test_init)
            
            
    # ===============================  Normalize, Power Transform, Split to Squ =================================
    def normalize_X(self, scaler=None):
        """
        normalizes data with scaler (default is StandardScaler)
        """
        
        self.scaler = StandardScaler() if scaler is None else scaler()
        
        self.X_train_normalized = self.scaler.fit_transform(self.X_train)
        self.X_test_normalized = self.scaler.transform(self.X_test)
                
        return True
    
    # ------------------------------------------------------------------------------------------------------------------#    
    def _unroll_XY_to_sequence(self, X, y):
        # src: https://analyticsindiamag.com/anomaly-detection-in-temperature-sensor-data-using-lstm-rnn-model/
        
        if len(X) != len(y):
            print(">>> X and y data should have same len...")
            return False
        
        list_X = []
        list_y = []
        
        for index in range(len(X) - self.training_seq_params['seq_len']):
            list_X.append(X[index: index + self.training_seq_params['seq_len']])
            list_y.append(y[index + self.training_seq_params['seq_len']])
        
        return np.asarray(list_X), np.asarray(list_y)

    # ------------------------------------------------------------------------------------------------------------------#
    def unroll_train_test_to_sequences(self):
        """
        Splits X_normalized to sequences
        >>> Example: [1,2,3,4,5] n_steps/ sequence_len=3 --> [1,2,3], [2,3,4], [3,4,5]
        """
        self.X_train_unrolled, self.y_train_unrolled = self._unroll_XY_to_sequence(
            X=self.X_train_normalized, 
            y=self.y_train.values)        
        self.X_train_shape.insert(0, self.X_train_unrolled.shape[0])
        
        self.X_test_unrolled, self.y_test_unrolled = self._unroll_XY_to_sequence(
            X=self.X_test_normalized, 
            y=self.y_test.values)
        self.X_test_shape.insert(0, self.X_test_unrolled.shape[0])
        # ------------------------------------------------------------------------------------------------------------------#
    # TODO: PowerTransform

    # ===========================================  Model   =================================================
    def set_train_test_data_shapes(self, verbose=True, shape_kwargs=None):
            """
            sets 
                self.training_seq_params 

            :param X: train or test; link to the data collection
            :paaram **shape_kwargs:
                >> seq_len
                !! n_features sets here from self.X_train --> so all EDA and FeatureSelection is done before
            """
            self.training_seq_params = shape_kwargs
            self.training_seq_params['n_features'] = len([c for c in self.X_train.columns if "Price" not in c])

            for shape in [self.X_train_shape, self.X_test_shape]:
                shape.append(self.training_seq_params['seq_len'])
                shape.append(self.training_seq_params['n_features'])
            

            if verbose:
                print('self.training_seq_params --> ', self.training_seq_params)
                print('self.X_train_shape --> ', self.X_train_shape)
                print('self.X_test_shape --> ', self.X_test_shape)

            return True
    
    # ------------------------------------------------------------------------------------------------------------------#
    def model_combine(self, template:list, compile_model=True, compile_dict=None, metrics=None, verbose=True):
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
        :param metrics: list of metrics that will be used in model compilation
        """
        
        self.model = keras.Sequential()
        for layer in template:                
            self.model.add(layer)
        
        if metrics is None:
            metrics = ['mae']
            
        if compile_model and (compile_dict is None):
            compile_dict = dict()
            compile_defaults = dict(optimizer='adam', loss='rmse', metrics=['mae'])            
            self._helpers_set_dict_default(compile_dict, compile_defaults)
            
        
        self.model.compile(**compile_dict)
        print(">>> model compiled")
        
        if verbose:
            self.model.summary()
        
        return True
    
    
    # ------------------------------------------------------------------------------------------------------------------#
    def model_fit(self, n_epoch=None, batch_size=32, verbose=2, early_stopping=True,
                  print_charts=True, use_tensorboard=False, return_results=False):
        """
        Fits the self.model, plots dynamics
        uses self.Xy_traintest_unrolled as input data for model        
        
        """
        if self.model is None:
            print(">>>No model detected. First you have to combine it...")
            return False
        if n_epoch is None:
            n_epoch = 10
        if (self.X_train_shape is None) or (self.X_test_shape is None):
            print(">>> before fitting set shapes with self.set_train_test_data_shapes()")
            return False

        n_features = self.X_train.shape[1]
        
        x_train = self.X_train_unrolled 
        x_train = x_train.reshape(*self.X_train_shape)
        y_train = self.y_train_unrolled

        x_test = self.X_test_unrolled 
        x_test = x_test.reshape(*self.X_test_shape)
        y_test = self.y_test_unrolled
        
        ES_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3) 
        fit_params = dict(
                            x=x_train, y=y_train,
                            epochs=n_epoch,
#                             batch_size=batch_size,
                            validation_data=(x_test, y_test),
                            verbose=verbose,
                            callbacks=[ES_callback] 
        )
        
        if not early_stopping:
            fit_params.pop('callbacks')
            
        results = self.model.fit(**fit_params)
        
        if (print_charts and use_tensorboard):
            # TODO
            pass
        
        if print_charts and (not use_tensorboard):
            # simple matplot chart
            loss = np.array(results.history['loss']);       
            val_loss = np.array(results.history['val_loss'])
            
            ax_properties = dict(ch_title=f'Loss by Epoch [{n_epoch}], EarlyStopping={early_stopping}\n', 
                                     x_label='epochs', 
                                     y_label='Loss', 
                                     x_lim=None, 
                                     y_lim=None, 
                                     label_series_base=val_loss, 
                                     shift=+5e-4)
                        
            
            fig, ax = plt.subplots(1,1, figsize=(15,5))
            ax.plot(loss, label="train_loss", marker='o', linewidth=1) 
            ax.plot(val_loss, label="test_loss", marker='o', linewidth=1)
                        
            self._helpers_plt_set_ax_properties(ax=ax, ax_properties=ax_properties)
            ax.legend(loc='upper right')
            
            ehpochs_before_stop = len(loss)
            ax.text(x=ehpochs_before_stop, y=loss[-1], s=f'{loss[-1]:.1f}')
            ax.text(x=ehpochs_before_stop, y=val_loss[-1], s=f'{val_loss[-1]:.1f}')
        
        
        if return_results:
            return results
        
        return True
    
    # ------------------------------------------------------------------------------------------------------------------#
    def model_predict(self, X):
        
        return self.model.predict(X)
    
    # ========================================  Plotting  ==============================================

    def plot_predicted_vs_test_price(self, **kwargs):
        """
        Plots predicted price for test part of the data 
        """
        
        test_y = pd.DataFrame(data=self.y_test_unrolled, index=self.y_test[self.training_seq_params['seq_len']:].index)        
        
        if self.y_pred is None:
            self.y_pred = pd.DataFrame(index=test_y.index, data=[
                    self.model_predict(seq.reshape(1,
                                                   self.training_seq_params['seq_len'],
                                                   self.training_seq_params['n_features']))[0][0] 
                                    for seq in self.X_test_unrolled], 
                        )
        
        
        fig, ax = plt.subplots(1, figsize=(15,5))
        
        self.plt_plot_ts(self.y_pred, label="predicted_price", ax=ax, color="#ffa64d", linewidth=2, **kwargs)
        self.plt_plot_ts(test_y, label="test_price", ax=ax, linewidth=1, **kwargs)
        
        plt.legend(['predicted_price','real_test_price'])
        plt.legend(loc='upper left')
        
    # ------------------------------------------------------------------------------------------------------------------#
    def plt_plot_ts(self, series: str, **kwargs):
        """
        Plots a timeseries plot for one feature

        :param series_name: series_name of DF
        :param kwargs: all other optional params:        
            >> :param titles: titles={'xlabel':xlabel, 'ylabel':ylabel, 'title':title, 'title_loc':title_loc}
                    >> title_loc --> location of the title
                    >> xy --> location of the title in [x,y] form. Overrides title_loc param
            >> :param color: line color "#8591e0"
            >> :param ls: line style "-'
            >> :param figsize: default figs_size=(10, 4)
            >> :param rc: rc dict for sns styling the chart def: {"grid.linewidth": 1, }
            >> :param ax: axes to plot timeseries.
            >> :param filters: array of filters. Exmpl: ["salary=>2.5e4", "salary <= 1e5"]
            >> :param annotate: if to show min max last on the chart

        :return:
        """
        # get defaults from **kwargs
        titles = kwargs.pop('titles') if 'titles' in kwargs else dict(xy=[1, 1.1], title_loc='center')
        color = kwargs.pop('color') if 'color' in kwargs else "#8591e0"
        ls = kwargs.pop('ls') if 'ls' in kwargs else "-"
        figsize = kwargs.pop('figsize') if 'figsize' in kwargs else (10, 4)
        rc = kwargs.pop('rc') if 'rc' in kwargs else {"grid.linewidth": 1, }
        ax = kwargs.pop('ax') if 'ax' in kwargs else plt.subplots(figsize=figsize)[1]
        filters = kwargs.pop('filters') if 'filters' in kwargs else None
        annotate = kwargs.pop('annotate') if 'annotate' in kwargs else None

        self.helpers_set_dict_default(titles, ['xlabel','ylabel','title','fontsize','xy','title_loc'])

        # prepaing Series
        titles["title"] = f"Predicted price vs Test price"

        if filters is not None:
            series, query_str = self.helpers_query_series(series, filters=filters)            
            filters_repr = self.helpers_filters_repr(query_str)
            titles["title"] = str(titles["title"]) + f"\nfilters: {filters_repr}"

        # plt
        ax.plot(series, color=color, ls=ls, **kwargs)
        ax.set_xlabel(titles['xlabel'])
        ax.set_ylabel(titles['ylabel'])

        # print('titles[xy] >>>> ', titles['xy'])
        if titles['xy'] is None:
            ax.set_title(titles['title'],
                         fontweight="bold", fontsize=titles['fontsize'], loc=titles['title_loc'])
        else:
            ax.set_title(titles['title'],
                         fontweight="bold", fontsize=titles['fontsize'], x=titles['xy'][0], y=titles['xy'][1])

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False);

        # annotate
        if annotate:
            last_point_ann = 'last'
            max_ann = 'max'
            min_ann = 'min'
            if series[0] == series.min():           
                last_point_ann = 'last, min'
                min_ann = ''
            if series[0] == series.max():           
                last_point_ann = 'last, max'
                max_ann = ''

            annotations = [(series.idxmin(), series.min(), min_ann), 
                           (series.idxmax(), series.max(), max_ann),
                           (series.index[0], series[0], last_point_ann)]

            for ann in annotations:
                ax.annotate(f'{ann[1]:,} ({ann[2]})', (mdates.date2num(ann[0]), ann[1]))

        # applying styling
        sns.set_context("poster", font_scale=.6, rc={"grid.linewidth": 1})

        # styling
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # gc
        series = None

        gc.collect()

    # ========================================  Helpers  ==============================================
    @staticmethod
    def _helpers_set_dict_default(dictionar, keys):
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
    # ------------------------------------------------------------------------------------------------------------------#                
    @staticmethod
    def _helpers_plt_set_ax_properties(ax=None, ax_properties=None):
        """
        formats plot created on the base of plt.subplots() with the properties in ax_properties
        :param ax:
        :param ax_properties:
        ax_properties = {
            0: dict(ch_title='Accuracy by Epoch\n', x_label='', y_label='Accuracy', x_lim=None, y_lim=977e-3,
                label_series_base=val_accur, shift=-3e-4),
            1: dict(ch_title='Loss by Epoch\n', x_label='epochs', y_label='Loss', x_lim=None, y_lim=None,
                label_series_base=val_loss, shift=+5e-4)
        }
        :return:
        """

        
        # general
        ax.legend(loc=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False);
        ax.grid(True, color='0.90')
  
        # ax_properties        
#         for i, val in enumerate(ax_properties['label_series_base']):
#                 ax.text(i, val + ax_properties['shift'], f"{val:.1f}", rotation=45)
                
        
        ax.set_ylim(ax_properties['y_lim'])
        ax.set_title(ax_properties['ch_title'], fontweight='bold')  # oc='left')
        ax.set_xlabel(ax_properties['x_label'])
        ax.set_ylabel(ax_properties['y_label'])
        
    # ------------------------------------------------------------------------------------------------------------------#
    # ===============================  Helpers from data class  ==============================================
    @staticmethod
    def helpers_query_series(series=None, merge_condition='and', filters=None, return_query_str=True):
        """
        filters series. If series_name, this series will be returned after filtering
        """
        query = ''
        for filt in filters:
            query += f'{filt} {merge_condition} '

        query = query.rsplit(' ', 2)[0]
        return (series.query(query), query)
    # ------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def helpers_set_dict_default(dictionar, keys, return_dict=False):
        """
        Sets values for keys in dict.
        
        :param dictionar: dictionary that has to be inspected for presence of keys
        :param keys:
            >> list: keys from list that are not found in dictionar, will be added with None value
            >> dict: keys from keys dict that are not found in dictionar, will be added with keys[key] value
                
        
        returns: None or dict 
        as dict is passed by ref, no real need for return
        
        """
        if isinstance(keys, list):
            for key in keys:
                dictionar[key] = dictionar[key] if key in dictionar else None
        
        if isinstance(keys, dict):
            for key, val in keys.items():
                dictionar[key] = dictionar[key] if key in dictionar else val
            
        if return_dict:
            return dictionar
    
    # ------------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def helpers_combine_features(df, features_to_plt, features_skip):
        """
        combines features on the base of include~ exclude~ features lists
        >> Protected -- if inheritance
        """
        feat_to_plt = []

        if features_to_plt is not None:
            feat_to_plt = features_to_plt

        if features_skip is not None:
            feat_to_plt = [f for f in df.columns if f not in features_skip]

        if len(feat_to_plt) == 0:
            feat_to_plt = df.columns
        
        return feat_to_plt
    
    @staticmethod
    def helpers_filters_repr(query: str):
        """
        takes query string and splits it to different lines with "and" separator
        """
        return '\n'.join(query.split("and"))
    
