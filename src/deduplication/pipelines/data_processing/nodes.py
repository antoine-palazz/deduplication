"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

# import numpy as np
import pandas as pd
import re
# import string
from tqdm import tqdm
from unidecode import unidecode
import warnings

tqdm.pandas()
warnings.filterwarnings('ignore')


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    return data.fillna("")


def normalize_strings(
    data: pd.DataFrame,
    str_cols: list
) -> pd.DataFrame:

    data[str_cols] = data[str_cols].progress_apply(
        lambda x: x.str.replace(r'\W', ' ').apply(
            lambda x: unidecode(re.sub(' +', ' ', x))
        ).str.strip().str.lower()
    )
    return data


def create_concatenated_column(
    data: pd.DataFrame,
    str_cols: list,
    concatenated_col_name: str
) -> pd.DataFrame:

    data[concatenated_col_name] = data[str_cols[0]]
    for col in str_cols[1:]:
        data[concatenated_col_name] += ' ' + data[col]

    return data


def preprocess_data(
    data: pd.DataFrame,
    str_cols: list = ['title', 'company_name', 'location', 'description'],
    concatenated_col_name: str = 'text'
) -> pd.DataFrame:

    data = remove_nans(data)
    data = normalize_strings(data, str_cols)
    data = create_concatenated_column(data, str_cols, concatenated_col_name)

    return data
