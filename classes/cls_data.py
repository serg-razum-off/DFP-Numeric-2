import pandas as pd
import os
import gc


class NumData:
    """
    SR: reads the num files related to trades of bitcoin
    """

    def __init__(self, dir_path, btc_exch_rates_filename, btc_total_circ_filename):
        """
        inits the data class
        :param dir_path: path to csvs
        :param btc_exch_rates_filename: name of file with btc_quotes
        :param btc_total_circ_filename: name of file with btc in circulation
        """
        self._data_btc = None
        self.clean_btc_exch_rates_data(dir_path, btc_exch_rates_filename)

    def clean_btc_exch_rates_data(self, dir_path, file_name):
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

        self._data_btc = btc_exch_raw
        del btc_exch_raw
        gc.collect()

    def merge_btc_exch_rates_w_btc_in_circulation(self, dir_path, file_name):
        btc_in_circ_raw = pd.read_csv(os.path.join(dir_path, file_name))
        btc_in_circ_raw['Data'] = pd.to_datetime(btc_in_circ_raw['Date'])
        btc_in_circ_raw['Value'] = btc_in_circ_raw['Value'].astype(float)
        btc_in_circ_raw['Btc_Mined'] = -btc_in_circ_raw['Value'].diff()
        
