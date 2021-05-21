import numpy as np
import pandas as pd
import os


class NumData:
    """
    SR: reads the num files related to trades of bitcoin
    """

    def __init__(self, dir_path, btc_quotes_filename):
        """
        inits the data class
        :param dir_path: path to csvs
        :param btc_quotes_filename: name of file with btc_quotes
        """
        self._data_btc_quotes_raw = (
            pd.read_csv(os.path.join(dir_path, btc_quotes_filename))
                .rename(columns={"Vol.": "Vol"}, inplace=False)
        )

    # def
