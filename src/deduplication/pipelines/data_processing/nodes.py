"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.6
"""

import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
tqdm.pandas()
from unidecode import unidecode


def remove_nans(data: pd.DataFrame) -> pd.DataFrame:
    return data.fillna("")


def normalize_strings(
    data: pd.DataFrame,
    str_columns: list
) -> pd.DataFrame:
    
    data[str_columns] = data[str_columns].progress_apply(
        lambda x: x.str.replace(r'\W', ' ').apply(
            lambda x: unidecode(re.sub(' +', ' ', x))
        ).str.strip().str.lower()
    )
    return data


def create_concatenated_column(
    data: pd.DataFrame,
    str_columns: list,
    concatenated_col_name: str
) -> pd.DataFrame:
    
    data[concatenated_col_name] = data[str_columns[0]]
    for col in str_columns[1:]:
        data[concatenated_col_name] += ' ' + data[col]
    
    return data


def preprocess_data(
    data: pd.DataFrame,
    str_columns: list = ['title', 'company_name', 'location', 'description'],
    concatenated_col_name: str = 'text'
) -> pd.DataFrame:

    data = remove_nans(data)
    data = normalize_strings(data, str_columns)
    data = create_concatenated_column(data, str_columns, concatenated_col_name)
    
    return data
