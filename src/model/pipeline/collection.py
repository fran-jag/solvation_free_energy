"""
Module for data collection. Configure data path in config/.env
"""

import pandas as pd
from loguru import logger

from config import model_settings


def load_data_from_csv(path=model_settings.data_path) -> pd.DataFrame:
    """
    Read data from csv in specified path.
    """
    logger.info("Reading file from: {}".format(path))
    df = pd.read_csv(path)
    return df


if __name__ == '__main__':
    df = load_data_from_csv()
    print(df.head())
