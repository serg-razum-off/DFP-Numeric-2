import pandas as pd
import os
import gc


class NumData:
    """
    SR: reads the num files related to trades of bitcoin
    """

    def __init__(self, dir_path, btc_exch_rates_filename):
        """
        inits the data class
        :param dir_path: path to csvs
        :param btc_quotes_filename: name of file with btc_quotes
        """
        self._data_btc_exch_rates = None
        self.clean_btc_exch_rates_data(dir_path, btc_exch_rates_filename)

    def clean_btc_exch_rates_data(self, dir_path, btc_exch_rates_filename):
        # reading exch_rate csv
        raw_data = (
            pd.read_csv(os.path.join(dir_path, btc_exch_rates_filename))
                .rename(columns={"Vol.": "Vol", "Change %": "Growth"}, inplace=False)
        )
        # converting ['Vol'] fom 5M into 5e6
        # src: https://stackoverflow.com/questions/39684548/convert-the-string-2
        # -90k-to-2900-or-5-2m-to-5200000-in-pandas-dataframe
        if any([x in raw_data['Vol'][0].upper() for x in ["K", "M"]]):  # if data in format of "K,M"
            raw_data['Vol'] = (
                    raw_data['Vol'].replace('-', '0')
                    .replace(r'[KM]+$', '', regex=True).astype(float)
                    *
                    raw_data['Vol'].str.extract(r'[\d\.]+([KM]+)', expand=False)
                    .fillna(1)
                    .replace(['K', 'M'], [1e3, 1e6]).astype(int)
            )

        # converting str repr of floats into floats
        cols = ('Price', 'Open', 'High', 'Low')
        if isinstance(raw_data[cols[0]][0], str):
            # tmp = raw_data.loc[:, cols].copy()
            raw_data.loc[:, cols] = (
                raw_data.loc[:, cols]
                    .apply(lambda x: x.str.replace(',', '').astype(float))
            )
        # converting ['Date'] str to dates
        raw_data.loc[:, 'Date'] = pd.to_datetime(raw_data['Date'])

        # converting ['Change %'] to float
        raw_data['Growth'] = raw_data['Growth'].str.replace('%', '').astype(
            float) / 100

        self._data_btc_exch_rates = raw_data
        del raw_data
        gc.collect()
