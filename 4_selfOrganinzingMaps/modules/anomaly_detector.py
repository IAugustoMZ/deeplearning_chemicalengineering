import numpy as np
import pandas as pd
import joblib as jb
import seaborn as sns
import scipy.stats as sts
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

RANDOM_SEED = 2
sns.set_style('darkgrid')

class AnomalyDetectionKit:

    def __init__(self) -> None:
        """
        class used for detecting anomalies from a dataset
        """
        pass

    def parallel_analysis(self,
                          data: pd.DataFrame,
                          plot: bool=False) -> None:
        """
        applies the Parallel Analysis Method to identify the recommended
        number of principal components to hold

        Parameters
        ----------
        data : pd.DataFrame
            data set of measurements
        plot : bool, optional
            flag to indicate the plot, by default False
        """
        # create two pipelines to apply the parallel analysis
        # method
        self.pca_complete = Pipeline([
            ('scaler', RobustScaler()),
            ('pca', PCA(random_state=RANDOM_SEED))
        ])
        pca_fake = Pipeline([
            ('scaler', RobustScaler()),
            ('pca', PCA(random_state=RANDOM_SEED))
        ])

        # create a fake copy of the inputted data
        data_fake = data.copy()
        n = data.shape[0]
        for col in data.columns:

            # calculate statistics
            avg = data[col].mean()
            sd = data[col].std()

            data_fake[col] = np.random.normal(
                loc=avg,
                scale=sd,
                size=n
            )

        # aply PCA in both datasets
        self.pca_complete.fit(data)
        pca_fake.fit(data_fake)

        # extract variance eigenvalues
        self.lambdas = self.pca_complete['pca'].explained_variance_
        lambdas_fake = pca_fake['pca'].explained_variance_

        # check where the fake surpasses the original
        self.p = np.where(lambdas_fake > self.lambdas)[0][0]+1

        print('Parallel Method Results')
        print('-'*50)
        print('Number of PCs retained: {}'.format(self.p))
        print('Explained Variance: {} %'.format(round(self.pca_complete['pca'].
                                                explained_variance_ratio_.cumsum()[self.p-1] *100, 2)))
        
        cols = [f'PC{k+1}' for k in range(data.shape[1])]
        if plot:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(cols[:self.p+10], self.lambdas[:self.p+10], 'ko', label='Original Data')
            ax.plot(cols[:self.p+10], lambdas_fake[:self.p+10], 'rx', label='Simulated Data')
            ax.set_xlabel('Principal Components', size=24)
            ax.set_ylabel('Eigenvalues', size=24)
            ax.tick_params(axis='x', labelsize=20, rotation=90)
            ax.tick_params(axis='y', labelsize=20)
            ax.legend(loc='best', prop={'size': 20}, facecolor='white')
            ax.set_title('Principal Components Selection - Parallel Analysis', size=28)
            plt.savefig('../8_imgs/som_studies/parallel_methods=.pdf', dpi=300, bbox_inches='tight')

    def fit_final(self,
                  data: pd.DataFrame,
                  n: int=-1) -> None:
        """
        fits the definitive model in to
        detect anomalies

        Parameters
        ----------
        data : pd.DataFrame
            data to fit the model
        n : int, optional
            number of desired components, by default -1
            if -1, the suggestion of parallel method wil be used
        """
        if ((n != -1) & (n != self.p)):

            self.p = n

            print('Number of PCs changed')
            print('-'*50)
            print('Number of PCs retained: {}'.format(self.p))
            print('Explained Variance: {} %'.format(round(self.pca_complete['pca'].
                                                    explained_variance_ratio_.cumsum()[self.p-1] *100, 2)))
            
        # fit pca
        self.final_pca = Pipeline([
            ('scaler', RobustScaler()),
            ('pca', PCA(n_components=self.p, random_state=RANDOM_SEED))
        ])
        self.final_pca.fit(data)

        # store important variables
        self.lambdas_final = self.final_pca['pca'].explained_variance_
        self.W = self.final_pca['pca'].components_.T

    def save_model(self,
                   model: object,
                   model_name: str) -> None:
        """
        save the model with the desired name

        Parameters
        ----------
        model : object
            model object
        model_name : str
            name of the model
        """
        jb.dump(
            model,
            '../1_models/som_studies/{}_{}_pcs.m'.format(model_name, self.p)
        )

    def calculate_T2(self,
                     data: pd.DataFrame,
                     alpha: float=0.05) -> tuple:
        """
        calculates the array of Hotelling's T2
        statistics and its respective confidence value

        Parameters
        ----------
        data : pd.DataFrame
            data to calculate T2
        alpha : float, optional
            significance level, by default 0.05

        Returns
        -------
        tuple
            array of T2 and confidence value
        """
        # calculate the array of scaled values
        x_scaled = self.final_pca['scaler'].transform(data)

        # calculate the array of T2
        T2s = np.array([
            xi.dot(self.W).dot(np.diag(self.lambdas_final**(-1)))
            .dot(self.W.T).dot(xi.T) for xi in x_scaled
        ])

        # calculate the confidence level of T2s
        T2max = self.confidence_ht2(n=data.shape[0], alpha=alpha)

        return T2s, T2max

    def confidence_ht2(self,
                       n: int,
                       alpha: float) -> float:
        """
        calculates the confidence level of Hotellings
        T2

        Parameters
        ----------
        n : int
            number of data rows
        alpha : float
            significance level

        Returns
        -------
        float
            critical value of the Hotellings
            T2s
        """
        # calculate the correction factor
        factor = (self.p*(n-1)) / (n - self.p)

        # calculate the critical value of HT2
        return factor * sts.f.ppf(1-alpha, self.p, n-self.p)