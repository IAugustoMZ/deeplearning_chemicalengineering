import numpy as np
import pandas as pd
from .minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

class SOM_outlier_detector:

    def __init__(self) -> None:
        """
        creates an outlier detector based on
        SOM (Self-Organizing Maps) neural networks
        """
        pass

    def fit_som(self,
                x: pd.DataFrame,
                n: int,
                sigma: int=1,
                learning_rate: float=0.5) -> None:
        """
        fits the self organizing map to detect
        outliers

        Parameters
        ----------
        x : pd.DataFrame
            data to detect outliers in
        n : int
            number of neurons used in each
            side of the SOM
        sigma : int, optional
            update neighborhood, by default 1
        learning_rate : float, optional
            learning rate, by default 0.5
        """
        # feature scaling
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(x)

        x_scaled = self.scaler.transform(x)

        # fitting the SOM
        self.som = MiniSom(
            x=n, y=n,
            input_len=x_scaled.shape[1],
            sigma=sigma,
            learning_rate=learning_rate
        )
        self.som.random_weights_init(x_scaled)
        self.som.train_random(
            x_scaled,
            num_iteration=1000
        )

    def identify_anomalies(self,
                           x: pd.DataFrame,
                           min_d: float=0.5) -> np.array:
        """
        identify outliers based on the average internode
        distance of the trained SOM        

        Parameters
        ----------
        x : pd.DataFrame
            data to be analyzed
        min_d : float, optional
            minimum distance to consider for potential
            outliers, by default 0.5

        Returns
        -------
        np.array
            indexes of potential outliers
        """
        # get normalized values
        x_scaled = self.scaler.transform(x)

        # find potential outliers nodes
        outliers_nodes = np.where(self.som.distance_map() > min_d)

        # map each data point to its node
        mappings = self.som.win_map(x_scaled)
        