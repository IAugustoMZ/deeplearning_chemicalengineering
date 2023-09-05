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

        # get the list of potential anomalies
        anom_list = []
        for i, j in zip(outliers_nodes[0], outliers_nodes[1]):

            # get the list of points mapped to the selected nodes
            anom_list.append(mappings[(i, j)])

        # clean empty values from the list
        anom_list = [x for x in anom_list if x != []]

        # concatenate all arrays in a single
        outliers = np.concatenate(anom_list, axis=0)

        # create a dataframe with a key to identify
        outliers = pd.DataFrame(outliers, columns=x.columns)
        outliers = outliers.round(6)
        outliers['key'] = outliers.apply(lambda row: ''.join(map(str, row)), axis=1)

        # get the scaled data as dataframe and create the keys
        x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
        x_scaled = x_scaled.round(6)
        x_scaled['key'] = x_scaled.apply(lambda row: ''.join(map(str, row)), axis=1)

        # find the data which corresponds to outliers
        outliers_idx = []
        for key in outliers['key'].values:
            outliers_idx.append(np.where(x_scaled['key'].values == key)[0][0])

        return outliers_idx
        