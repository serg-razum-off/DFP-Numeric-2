import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import glob


class DataManager:
    """
    SR: reads the files related to trades of bitcoin and preprocesses them
    """

    def __init__(self, dir_path: str, btc_exch_rates_filename: str):
        """
        inits the DataManager class

        :param dir_path: path to csvs
        :param btc_exch_rates_filename: name of file with btc_quotes
        :param btc_other_filenames: list of filenames that store additional data
                                    All have to be in same format: dateCol, ValCol
        """
        self.data_btc = None
        self.data_files = [f for f in filter(os.path.isfile, os.listdir(dir_path))]
        self.data_files_prop = pd.read_csv(os.path.join(dir_path, "files_properties.csv"))
        self._btc_exch_rates_filename = btc_exch_rates_filename
        self._set_self_data_btc(dir_path, btc_exch_rates_filename)
        self._self_data_btc_Add_other_metrics(dir_path=dir_path)

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
                curr_file['Btc_Mined'] = -curr_file['Value'].diff()
                curr_file['Btc_Mined'].fillna(method='bfill', inplace=True)  # filling last day (Nan) as one before

            curr_file.set_index('Date', inplace=True)
            # rename Value col to the core part of filename
            curr_file.rename(columns={"Value": file_names[i].split('-')[-1].split(".")[0]},
                             inplace=True)

            self.data_btc = self.data_btc.merge(curr_file, right_index=True, left_index=True)
            
        #check if all files from data folder are included to files_properties.csv
        files_in_filesproperties = self.data_files_prop["FileName"].dropna().to_list()
        table_columns = self.data_btc.columns.to_list()
            
        missed_files = [f for f in  
                            [col for col in table_columns if col not in ['Price', 'Open', 'High', 'Low', 'Vol', 'Growth', 'Btc_Mined']]
                        if not any(f in s for s in files_in_filesproperties)]

        if len(missed_files) > 0:
            print("!!! add to files_properties descr of files with these columns: ")
            print(missed_files)

    def _plt_plot_timeseries(self, series: str, titles=None, color="#8591e0", ls="-", figsize=(10, 4), rc=None,
                             ax=None):
        """
        Plots a timeseries plot

        :param series: Name of the (Series that has index of datetime type)
        :param titles: titles={'xlabel':xlabel, 'ylabel':ylabel, 'title':title}
        :param color: line color "#8591e0"
        :param ls: line style "-'
        :param figsize: default figsize=(10, 4)
        :param rc: rc dict for sns styling the chart def: {"grid.linewidth": 1, }
        :param ax: axes to plot timeseries.
        :return:
        """

        if rc is None:
            rc = {"grid.linewidth": 1, }

        if titles is None:
            titles = {}

        titles['xlabel'] = titles['xlabel'] if 'xlabel' in titles else None
        titles['ylabel'] = titles['ylabel'] if 'ylabel' in titles else None
        titles['title'] = titles['title'] if 'title' in titles else None

        if ax is None:
            ax = plt.subplots(figsize=figsize)[1]

        # plt
        ax.plot(self.data_btc[series], color=color, ls=ls)
        ax.set_xlabel(titles['xlabel'])
        ax.set_ylabel(titles['ylabel'])
        ax.set_title(titles['title'], fontweight="bold")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False);

        # applying styling
        sns.set_context("poster", font_scale=.6, rc={"grid.linewidth": 1})

        # styling
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    def preview_features(self, features_to_plt=None, features_skip=None, fig_size=(25, 20),
                         fig_title="Previewing BTC data"):
        """
        plots specified features as timeseries 

        :param features_to_plt: if some features are to be plot
        :param features_skip: if some features are to be skipped
        :param fig_size: figsize of the combined charts -- tuple
        :param fig_title: title of the combined figure
        :return: None
        """
        feat_to_plt = []

        if features_to_plt is not None:
            feat_to_plt = [f for f in self.data_btc.columns if f in features_to_plt]

        if features_skip is not None:
            feat_to_plt = [f for f in self.data_btc.columns if f not in features_skip]

        if len(feat_to_plt) == 0:
            feat_to_plt = self.data_btc.columns

        fig, ax = plt.subplots(len(feat_to_plt), 1)
        fig.suptitle(fig_title, fontsize=17, fontweight='bold')

        for i, feature in enumerate(feat_to_plt):
            self._plt_plot_timeseries(feature, figsize=(17, 5),
                                      titles=dict(title=feature), ax=ax[i], color="#8591e0")

        fig.set_size_inches(fig_size)
