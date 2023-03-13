import csv
from kedro.io import AbstractDataSet
import numpy as np
import pandas as pd


class BrokenCSVDataSet(AbstractDataSet[pd.DataFrame, pd.DataFrame]):

    def __init__(self, filepath: str):
        """Creates a new instance of BrokenCSVDataSet

        Args:
            filepath: The location of the file to load / save data.
        """
        self._filepath = filepath

    def _load(self) -> np.ndarray:
        """Loads data from the file.

        Returns:
            Data from the file as a Pandas dataframe.
        """
        with open(self._filepath, mode="rb") as file_in:
            df = pd.read_csv(file_in,
                             engine='c',
                             delimiter=',',
                             quotechar='"',
                             quoting=csv.QUOTE_ALL,
                             lineterminator='\n')
        print(f'{len(df)} rows in the dataset')
        return df

    def _save(self, df: pd.DataFrame) -> None:
        """Saves image data to the specified filepath"""
        df.to_csv(str(self._filepath))

    def _describe(self) -> dict:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(filepath=self._filepath)
