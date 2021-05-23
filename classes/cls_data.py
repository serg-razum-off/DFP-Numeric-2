import pandas as pd
import os
import gc


class DataManager:
    """
    SR: reads the num files related to trades of bitcoin
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
        if isinstance(btc_exch_raw[cols[0]][0], str):
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
        :param file_list: list of file_names that have to be extracted and added to exch rate
        """
        result = None
        file_list = [f for f in os.listdir(dir_path)
                     if (self._btc_exch_rates_filename not in f)
                     and ("propert" not in f)]

        for file_name in file_list:
            curr_file = pd.read_csv(os.path.join(dir_path, file_name))
            curr_file['Date'] = pd.to_datetime(curr_file['Date'])
            curr_file['Value'] = curr_file['Value'].astype(float)
            if 'TOTBC' in file_name:
                curr_file['Btc_Mined'] = -curr_file['Value'].diff()
                curr_file['Btc_Mined'].fillna(method='bfill', inplace=True)  # filling last day (Nan) as one before
            curr_file.set_index('Date', inplace=True)
            curr_file.rename(columns={"Value": file_name.split('-')[-1].split(".")[0]}, inplace=True) # rename Value col

            self.data_btc = self.data_btc.merge(curr_file, right_index=True, left_index=True)