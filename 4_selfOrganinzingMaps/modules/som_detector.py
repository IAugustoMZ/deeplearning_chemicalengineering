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
    
    def convergence_analysis(self, 
                             x: pd.DataFrame,
                             n:int,
                             n_soms: int,
                             eps:float=0.001) -> float:
        """
        performs a stabilization analysis to check if
        the majority of the readings achieved convergence
        in AID

        Parameters
        ----------
        x : pd.DataFrame
            dataframe to fit soms
        n : int
            number of iterations to perform
        n_soms : int
            square root of the number of neurons
        eps : float, optional
            tolerance limit for convergence
            qualification, by default 0.01

        Returns
        -------
        float
            percentage of converged readings
        """
        dists = []

        # make N fits and calculate AID for all samples
        for _ in range(n):

            # fit SOM
            self.fit_som(x=x, n=n_soms)

            # calculate AID
            dists.append(self.average_distances(x=x).values)

        # concatenate AIDs
        all_dists = np.concatenate(dists, axis=1)

        # create columns
        df_cols = [f'Run{r+1}' for r in range(all_dists.shape[1])]

        # create dataframe
        df_runs = pd.DataFrame(all_dists, columns=df_cols)

        # check convergence
        df_runs['convergence'] = df_runs.apply(self.check_convergence, eps=eps, axis=1)

        return df_runs['convergence'].mean()
    
    def check_convergence(self, row: pd.DataFrame, eps: float) -> int:
        """
        checks if the convergence occurred for a particular
        set of SOMs runs for a specific reading

        Parameters
        ----------
        row : pd.DataFrame
            row with readings
        eps : float
            tolerance for convergence

        Returns
        -------
        int
            1 - convergence / 0 - otherwise
        """

        # iterate through all readings to check convergence
        sample = pd.DataFrame(row.values, columns=['aid'])

        # calculate cumulated mean
        sample['cum_avg'] = sample['aid'].expanding().mean()

        # calculate rolling statistics of last 10 % period
        sample['rolling_std'] = sample['cum_avg'].rolling(5).std()
        sample['rolling_mean'] = sample['cum_avg'].rolling(5).mean()
        sample['rolling_cv'] = abs(sample['rolling_std'] / sample['rolling_mean'])

        # check last cv for convergence
        if (sample['rolling_cv'] < eps).values[-1]:
            return 1
        
        return 0
        