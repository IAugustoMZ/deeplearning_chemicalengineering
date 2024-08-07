import os
import json
import joblib
import numpy as np
import pandas as pd
from .bootstrap import *
from .minisom import MiniSom
from .forms import PlantDesignForm


# models path
MODELS_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)
                    )
                )
            )
        )
    ), '1_models', 'som_studies'
)
ACTUALS_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

class RiskAssessmentSOM:

    def __init__(self) -> None:
        """
        class to estimate the KPIs using the ML models
        """
        # load the trained scaler
        scaler_path = os.path.join(MODELS_PATH, 'scaler_som.pkl')
        self.scaler = joblib.load(scaler_path)

        # load the contract
        with open(os.path.join(ACTUALS_PATH, 'contract.json'), 'r') as f:
            self.contract = json.load(f)

        # fit the SOM model in the available data
        self.get_reference_data()
        self.fit_som()
        self.get_winning_nodes()

    def get_winning_nodes(self) -> None:
        """
        Get the winning nodes for the reference data
        """
        # get the winning nodes
        winning = []
        for i in range(self.reference_x.shape[0]):
            winning_node = self.som.winner(self.reference_x.values[i])
            winning.append(winning_node)

        # append to the data
        self.reference_data['winning_node'] = winning

    def fit_som(self) -> None:
        """
        Fit the SOM model
        """
        # get the size of the SOM
        som_size = int(np.sqrt(5 * np.sqrt(self.reference_x.shape[0])))

        # fit the SOM model
        self.som = MiniSom(
            x=som_size,
            y=som_size,
            input_len=self.reference_x.shape[1],
            sigma=1,
            learning_rate=0.5
        )
        self.som.random_weights_init(self.reference_x.values)
        self.som.train_random(self.reference_x.values, num_iteration=200)

        # get the distance map
        self.distance_map = self.som.distance_map()

    def get_reference_data(self) -> None:
        """
        Get the reference data for the SOM
        """
        # load the reference data
        reference_data_path = os.environ.get('SOM_PRETRAINED_DATA')
        self.reference_data = pd.read_csv(reference_data_path, index_col=[0])

        # get the x columns from the contract
        x_cols = self.contract['columns_mapping'].values()

        # select the x columns
        self.reference_x = self.reference_data[x_cols]

        # scale the reference data
        self.reference_x = pd.DataFrame(
            self.scaler.transform(self.reference_x),
            columns=self.reference_x.columns
        )

    def prepare_data(self, data: PlantDesignForm) -> None:
        """
        Prepare the data for the KPIs estimation

        Parameters
        ----------
        data : PlantDesignForm
            the form data
        """
        self.dict_data = {
            name: value.data for name, value in data.__dict__['_fields'].items() if
            name not in ['csrf_token', 'submit']
        }
        
        # Prepare the data for the KPIs estimation
        self.data = pd.DataFrame.from_dict(self.dict_data, orient='index').T
        
        # change the cols
        self.data = self.data.rename(columns=self.contract['columns_mapping'])

        # scale the data
        self.data = pd.DataFrame(
            self.scaler.transform(self.data),
            columns=self.data.columns
        )

    def get_winning_node_for_input(self) -> int:
        """
        Get the winning node for the input data
        """
        # get the winning node
        winning_node = self.som.winner(self.data.values[0])

        return winning_node
    
    def estimate_risk(self) -> float:
        """
        Estimate the risk of not achieving the goals

        Returns
        -------
        float
            the risk of not achieving the goals
        """
        # get the winning node
        winning_node = self.get_winning_node_for_input()

        # get the samples in the winning node
        samples = self.reference_data.loc[self.reference_data['winning_node'] == winning_node, ['msp', 'vpl']]

        # bootstrap the samples
        bootstrapped_MSP = bootstrap(samples['msp'], n_bootstraps=1000)
        bootstrapped_VPL = bootstrap(samples['vpl'], n_bootstraps=1000)

        # calculate the risk
        MSP_GOAL = self.contract['goalMSP']
        VPL_GOAL = self.contract['goalNPV']
        riskMSP = round(np.mean(bootstrapped_MSP > MSP_GOAL) * 100, 2)
        riskVPL = round(np.mean(bootstrapped_VPL < VPL_GOAL) * 100, 2)

        return {
            'MSP': riskMSP,
            'NPV': riskVPL
        }


    

        
        