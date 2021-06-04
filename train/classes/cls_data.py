import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import glob


class DataManager:
    """
    SR: reads the files related to trades of bitcoin and preprocesses them. 
        >> combining features from different src files
        >> nulls handling
        >> normalization
    """

    # ===============================  Init  ====================================================
    def __init__(self, dir_path: str, btc_exch_rates_filename: str):
        """
        inits the DataManager class

        :param dir_path: path to csvs
        :param btc_exch_rates_filename: name of file with btc_quotes
        :param btc_other_filenames: list of filenames that store additional data
                                    All have to be in same format: dateCol, ValCol
        """
        self.data_btc = None
        self.data_btc_normalized = None
        self.data_files = [f for f in filter(os.path.isfile, os.listdir(dir_path))]
        self.data_files_prop = pd.read_csv(os.path.join(dir_path, "files_properties.csv"))
        self._btc_exch_rates_filename = btc_exch_rates_filename
        self._set_self_data_btc(dir_path, btc_exch_rates_filename)
        self._self_data_btc_Add_other_metrics(dir_path=dir_path)

        # ===============================  Getting data from csv ===================================

    def _set_self_data_btc(self, dir_path, file_name):
        # reading exch_rate csv        
        btc_exch_raw = (
            pd.read_csv(os.path.join(dir_path, file_name))
                .rename(columns={"Vol.": "Vol", "Change %": "Growth"}, inplace=False)
        )
        # converting ['Vol'] fom 5M into 5e6
        # src: https://stackoverflow.com/questions/39684548/convert-the-string-2
        # -90k-to-2900-or-5-2m-to-5200000-in-pandas-dataframe
        if any([x in btc_exch_raw['Vol'][0].upper() for x in ["K", "M"]]):  # if data in format of "K,M"
            btc_exch_raw['Vol'] = (
                    btc_exch_raw['Vol'].replace('-', '0')
                    .replace(r'[KM]+$', '', regex=True).astype(float)
                    *
                    btc_exch_raw['Vol'].str.extract(r'[\d\.]+([KM]+)', expand=False)
                    .fillna(1)
                    .replace(['K', 'M'], [1e3, 1e6]).astype(int)
            )

        # converting str repr of floats into floats
        cols = ('Price', 'Open', 'High', 'Low')
        if btc_exch_raw[cols[0]].dtype != np.number:
            # tmp = btc_exch_raw.loc[:, cols].copy()
            btc_exch_raw.loc[:, cols] = (
                btc_exch_raw.loc[:, cols]
                    .apply(lambda x: x.str.replace(',', '').astype(float))
            )
        # converting ['Date'] str to dates
        btc_exch_raw.loc[:, 'Date'] = pd.to_datetime(btc_exch_raw['Date'])

        # converting ['Change %'] to float
        btc_exch_raw['Growth'] = btc_exch_raw['Growth'].str.replace('%', '').astype(
            float) / 100

        self.data_btc = btc_exch_raw.set_index("Date")
        del btc_exch_raw
        gc.collect()

    def _self_data_btc_Add_other_metrics(self, dir_path: str):
        """
        Gets CSV files (columns: date, value), merges them, adds to Exchange Rate

        :param dir_path: path to csv files        
        """
        result = None
        # getting paths of additional files (same shape) and their names
        files_paths = [f for f in glob.glob(os.path.join(dir_path, "*.csv"))
                       if (self._btc_exch_rates_filename not in f)
                       and ("propert" not in f)]
        file_names = list(map(lambda X: X.split('/')[-1], files_paths))

        for i, file_path in enumerate(files_paths):
            curr_file = pd.read_csv(file_path)
            curr_file['Date'] = pd.to_datetime(curr_file['Date'])
            curr_file['Value'] = curr_file['Value'].astype(float)
            if 'TOTBC' in file_path:
                curr_file['BTC_MINED'] = -curr_file['Value'].diff()
                curr_file['BTC_MINED'].fillna(method='bfill', inplace=True)  # filling last day (Nan) as one before

            curr_file.set_index('Date', inplace=True)
            # rename Value col to the core part of filename
            curr_file.rename(columns={"Value": file_names[i].split('-')[-1].split(".")[0]},
                             inplace=True)

            self.data_btc = self.data_btc.merge(curr_file, right_index=True, left_index=True, how='left') # leftjoin
        # gc
        curr_file = None
        gc.collect()

        # fillnulls forward. (if null happens is on the later date, fill it from the earlier date)
        # https://monosnap.com/file/rr8KGWAvsLyBJt6bxRhYUzIpX5OFBp
        self.data_btc.fillna(method="bfill", inplace=True)

        # check if all files from data folder are included to files_properties.csv
        files_in_filesproperties = self.data_files_prop["FileName"].dropna().to_list()
        table_columns = self.data_btc.columns.to_list()

        missed_files = [f for f in
                        [col for col in table_columns if
                         col not in ['Price', 'Open', 'High', 'Low', 'Vol', 'Growth', 'BTC_MINED']]
                        if not any(f in s for s in files_in_filesproperties)]

        if len(missed_files) > 0:
            print("!!! add to files_properties descr of files with these columns: ")
            print(missed_files)

    # ===============================  Norm, Saving results =======================================
    def save_combined_csv(self, path=None, datasets=None):
        """
        Saves prepared file to folder as a source for NN
        
        :param path: path to save csv file(s)
        :param datasets: list of names of the datasets to save. 
            >> Allowed values: ["data_btc", "data_btc_normalized"]. 
            >> Default: ["data_btc_normalized"]
        """
        if path is None:
            path = "../data/ready_to_train"
            if not os.path.exists(path):
                path = "."

        if datasets is None:
            datasets = ["data_btc_normalized"]

        if "data_btc" in datasets:
            self.data_btc.to_csv(os.path.join(path, "data_btc_combined.csv"))
            print(f">> self.data_btc --> saved to {path}")

        if "data_btc_normalized" in datasets:
            self.data_btc_normalized.to_csv(os.path.join(path, "data_btc_combined_nirmalized.csv"))
            print(f">> self.data_btc_normalized --> saved to {path}")

    # ===============================  Plotting area ==============================================
    
    def plt_plot_ts(self, series_name: str, **kwargs):
        """
        Plots a timeseries plot for one feature

        :param series_name: series_name of DF
        :param kwargs: all other optional params:        
            >> :param titles: titles={'xlabel':xlabel, 'ylabel':ylabel, 'title':title, 'title_loc':title_loc}
                    >> title_loc --> location of the title
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
        series = self.data_btc[series_name]
        titles["title"] = f"Time Series for [{series_name}]" if titles["title"] is None else titles["title"]
        
        if filters is not None:
            series, query_str = self.helpers_query_df(self.data_btc, series_name, filters=filters)            
            filters_repr = self.helpers_filters_repr(query_str)
            titles["title"] = str(titles["title"]) + f"\nfilters: {filters_repr}"

        # plt
        ax.plot(series, color=color, ls=ls)
        ax.set_xlabel(titles['xlabel'])
        ax.set_ylabel(titles['ylabel'])
        ax.set_title(titles['title'],
                     fontweight="bold", fontsize=titles['fontsize'], loc=titles['title_loc'],
                     x=titles['xy'][0], y=titles['xy'][1])
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

    # ------------------------------------------------------------------------------------------------------------------#
    def plt_plot_multiple_ts(self, features_to_plt=None, features_skip=None, **kwargs):
        """
        plots specified features as timeseries 

        :param features_to_plt: if some features are to be plot
        :param features_skip: if some features are to be skipped
        :param kwargs: every kwarg that is applicable for self.plt_plot_ts plus:        
            >> :param tot_figsize: tot_figsize of the combined charts -- tuple
            >> :param titles['fig_title']: title of the combined figure
            >> !!! param figsize for nested charts is redundant. Not to send with kwargs

        :return: None
        """
        
        titles = kwargs.pop('titles') if 'titles' in kwargs else dict()
        color = kwargs.pop('color') if 'color' in kwargs else "#8591e0"
        tot_figsize = kwargs.pop('tot_figsize') if 'tot_figsize' in kwargs else (25, 20)
        
        # setting defaults, if user didn't pass values
        self.helpers_set_dict_default(titles, dict(fig_title="Previewing BTC TimeSeries data", 
                                                   title_loc='right',
                                                   xy=titles['xy'] if 'xy' in titles else [1, 1.1],
                                                   xy_suptitle = titles['xy_suptitle'] if 'xy_suptitle' in titles else [0.5,.95]
                                                  ))
        
        feat_to_plt = self.helpers_combine_features(self.data_btc, features_to_plt, features_skip)

        fig, ax = plt.subplots(len(feat_to_plt), 1) 
        fig.suptitle(titles['fig_title'], x=titles['xy_suptitle'][0], y=titles['xy_suptitle'][1], fontsize=18, fontweight='bold')

        for i, feature in enumerate(feat_to_plt):
            titles['title'] = feature 
            curr_ax = ax[i] if len(feat_to_plt) > 1 else ax
            self.plt_plot_ts(feature, titles=titles,
                                      ax=curr_ax, color=color, **kwargs)

        fig.set_size_inches(tot_figsize)

    # ------------------------------------------------------------------------------------------------------------------#
    def plt_hist(self, series_name, annotate=True, return_details=False, **kwargs):
        """
        SR: plots histogram.
          Att!! sensitive to NaNs!! 
          if data[column].isna().sum() > 1:
            print(f"{column} has nulls: {data[column].isna().sum()}")
            continue
        :param series_name: Name of the series to plot        
        :param return_details: if True, (hist_txt, hist_arr) will be returned        
        :param annotate: if to annotate bars of histogram with numeric values
        
        :param kwargs: all other params
            >> :param x",y: will be aplied to position of the title
            >> :param bins": num of bins
            >> :param ann_fontsize": fontsize of the bins annotations
            >> :param filters: array of filters. Exmpl: ["salary=>2.5e4", "salary <= 1e5"]
            >> :param bins: int or list as per bins that have to be shown in hist. Default is Series.unique().__len__()
            >> :param titles: if None, dict will be applied: {'xlabel': None, 'ylabel': None, 'title': None}
            >> :param ax: if None, will be initialized to ax = plt.subplots(figsize=figsize)[1]
            >> :param figsize: default is figsize=(10, 4)
            
            
        :return: if return_details: return hist_txt, hist_arr
        """
        
        series = self.data_btc[series_name]
        
        # getting defaults from kwargs
        ax = kwargs.pop('ax') if 'ax' in kwargs else None
        filters = kwargs.pop('filters') if 'filters' in kwargs else None
        bins = kwargs.pop('bins') if 'bins' in kwargs else 15
        color = kwargs.pop('color') if 'color' in kwargs else '#537ddf'
        figsize = kwargs.pop('figsize') if 'figsize' in kwargs else (10, 4)        
        titles = kwargs.pop('titles') if 'titles' in kwargs else dict(title=series_name)

        self.helpers_set_dict_default(titles, ['xlabel', 'ylabel'])
        
        if filters is not None:
            series, query_str = self.helpers_query_df(self.data_btc, series_name, filters=filters)            
            filters_repr = self.helpers_filters_repr(query_str)
            titles["title"] = str(titles["title"]) + f"\nfilters: {filters_repr}"
        
        if ax is None:
            ax = plt.subplots(figsize=figsize)[1]
            fontsize = 12
        else:
            fontsize = 10
        
        if "ann_fontsize" in kwargs:
            ann_fontsize = kwargs.pop("ann_fontsize")
        else:
            ann_fontsize = int(fontsize * .9)
        
        # print("kwargs --> ", kwargs)
        # cann't send with kwargs as x is a positional arg of ax.hist()
        x_title = kwargs.pop('x') if 'x' in kwargs else .5 
        y_title = kwargs.pop('y') if 'y' in kwargs else 1 
        
        hist_arr = ax.hist(series, color=color, bins=bins, **kwargs)
        ax.set_xlabel(titles['xlabel'])

        ax.set_ylabel(titles['ylabel'])
        ax.set_title(titles['title'], fontweight="bold", fontsize=fontsize, x=x_title, y=y_title, **kwargs)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False);

        shift = 0.01

        # SR: explanatory collecting data to hist_txt list
        hist_txt = ['======= Chart details =======']
        if type(bins) == list:
            bins = len(bins) - 1
        for i in range(bins):
            if (bins < 10) and annotate:
                ax.text(hist_arr[1][i], hist_arr[0][i] + hist_arr[0][i] * shift, f"{hist_arr[0][i]:.0f}", 
                        weight='bold', fontsize=ann_fontsize)
            hist_txt.append(f"[{hist_arr[1][i]:,.1f} - {hist_arr[1][i + 1]:,.1f}] --> [{hist_arr[0][i]:,.1f}]")

        # SR: annotating bars
        if bins >= 10 and annotate:
            if (bins > 10) and (bins < 20):
                denum = 5
            else:
                denum = 10
            ax.text(hist_arr[1][0], hist_arr[0][0] + hist_arr[0][0] * shift, f"{hist_arr[0][0]:.0f}",
                    weight='bold', fontsize=ann_fontsize)  # start
            ax.text(hist_arr[1][-2], hist_arr[0][-1] + hist_arr[0][-1] * shift, f"{hist_arr[0][-1]:.0f}",
                    weight='bold', fontsize=ann_fontsize)  # end
            mx_idx = np.where(hist_arr[0] == hist_arr[0].max())[0][0]
            ax.text(hist_arr[1][mx_idx], hist_arr[0][mx_idx] + hist_arr[0][mx_idx] * shift, f"{hist_arr[0][mx_idx]:.0f}",
                    weight='bold', fontsize=ann_fontsize)  # mx

            for i in range(0, bins - 1, bins // denum):
                ax.text(hist_arr[1][i], hist_arr[0][i] + hist_arr[0][i] * shift, f"{hist_arr[0][i]:.0f}", 
                        weight='bold', fontsize=ann_fontsize)

        if return_details:
            return hist_txt, hist_arr

    # ------------------------------------------------------------------------------------------------------------------#
    def plt_hist_grid(self, features_to_plt=None, features_skip=None, suptitle=None, ncols=4, figsize=(20, 15), **kwargs):
        """

        :param suptitle:
        :param ncols:
        :param figsize: main canvas figsize
        :param kwargs: all other params that will be --> to plt_hist (bins, annotate...)
        :return:
        """
        
        feat_to_plt = self.helpers_combine_features(self.data_btc, features_to_plt, features_skip)
        #print(feat_to_plt)
        
        data = self.data_btc[feat_to_plt] # leaving only selected features

        if suptitle is None:
            suptitle = 'Data distribution Preview'

        whole_p = len(data.columns) // ncols
        nrws = whole_p if len(data.columns) % ncols == 0 else whole_p + 1
        fig, axes = plt.subplots(nrws, ncols, figsize=figsize)
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=.93)
                
        for i, column_name in enumerate(data.columns):
            
            chart_row = i // ncols
            chart_col = i - chart_row * 4 if whole_p > 1 else i
            ax = axes[chart_row][chart_col] if whole_p > 1 else axes[chart_col]
            # print("chart_row, chart_col, ax -->", chart_row, chart_col, ax)
            
            num_nulls = data[column_name].isna().sum()
            if num_nulls > 1:
                # print(f"{column}-{i} has NaNs: {data[column].isna().sum()}")
                data[column_name].hist(ax=ax, color='orange');
                ax.set_title(column_name + f"_nans [{num_nulls}]", color='#cc6666', fontweight='bold')

                continue
            # print("kwargs -->", kwargs)
            self.plt_hist(column_name, ax=ax, **kwargs)

    # ===============================  Helpers area ==============================================
    @staticmethod
    def helpers_query_df(df, series_name=None, merge_condition='and', filters=None, return_query_str=True):
        """
        filters DF. If series_name, this series will be returned after filtering
        """
        query = ''
        for filt in filters:
            query += f'{filt} {merge_condition} '

        query = query.rsplit(' ', 2)[0]
        return (df.query(query)[series_name], query) if series_name is not None else (df.query(query), query)
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
            feat_to_plt = [f for f in df if f in features_to_plt]

        if features_skip is not None:
            feat_to_plt = [f for f in df if f not in features_skip]

        if len(feat_to_plt) == 0:
            feat_to_plt = df.columns
        
        return feat_to_plt
    
    @staticmethod
    def helpers_filters_repr(query: str):
        """
        takes query string and splits it to different lines with "and" separator
        """
        return '\n'.join(query.split("and"))