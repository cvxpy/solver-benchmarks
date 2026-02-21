"""Download and prepare the bikeshare dataset."""

import pandas as pd
import numpy as np
import requests
from io import BytesIO
from zipfile import ZipFile
import scipy as sp


def load_bikeshare_data():
    """Load and return the bikeshare dataset from a public URL.

    Returns pandas DataFrame.
    """
    URL = "https://s3.amazonaws.com/capitalbikeshare-data/2011-capitalbikeshare-tripdata.zip"
    resp = requests.get(URL)
    resp.raise_for_status()

    with ZipFile(BytesIO(resp.content)) as zf:
        csv_name = "2011-capitalbikeshare-tripdata.csv"
        with zf.open(csv_name) as f:
            df_bikeshare = pd.read_csv(f)
    return df_bikeshare


def get_bikeshare_features(df_bikeshare):
    """Extract features from the bikeshare DataFrame.

    Returns:
        - A_sparse is a one-hot encoding of start and end station numbers.
        - A_dense is a dense matrix of sine and cosine features for the start time.
    """
    A_start_location = sp.sparse.csr_matrix(
        pd.get_dummies(df_bikeshare["Start station number"], sparse=True).values
    )
    A_end_location = sp.sparse.csr_matrix(
        pd.get_dummies(df_bikeshare["End station number"], sparse=True).values
    )
    A_sparse = sp.sparse.hstack([A_start_location, A_end_location])

    t = pd.to_datetime(df_bikeshare["Start date"])
    sec = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy()
    A_dense = np.array(
        [np.sin(2 * np.pi * sec / 86400), np.cos(2 * np.pi * sec / 86400)]
    ).T

    return A_sparse, A_dense
