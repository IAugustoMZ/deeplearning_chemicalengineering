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

    def average_distances(self,
                          x: pd.DataFrame) -> pd.DataFrame:
        """
        map the average distance from neighbor neurons
        for all rows        

        Parameters
        ----------
        x : pd.DataFrame
            data to be analyzed

        Returns
        -------
        pd.DataFrame
            dataframe with the average distance
        """
        # get normalized values
        x_scaled = pd.DataFrame(
            self.scaler.transform(x),
            columns=x.columns
        )

        # get all nodes list
        nodes = self.som.distance_map()

        # map each data point to its node
        x_scaled['dist'] = x_scaled.apply(
            lambda row: nodes[self.som.winner(row.values)], axis=1)

        return x_scaled[['dist']]
        