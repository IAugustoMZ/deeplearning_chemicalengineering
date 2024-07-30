import os
import json
import joblib
import pandas as pd
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

class KPIsEstimate:
    def __init__(self) -> None:
        """
        class to estimate the KPIs usinf the ML models
        """
        # Load the trained models
        msp_model_path = os.path.join(MODELS_PATH, 'model_msp.pkl')
        vpl_model_path = os.path.join(MODELS_PATH, 'model_vpl.pkl')
        self.msp_model = joblib.load(msp_model_path)
        self.vpl_model = joblib.load(vpl_model_path)

        # load the contract
        with open(os.path.join(ACTUALS_PATH, 'contract.json'), 'r') as f:
            self.contract = json.load(f)

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
        
    def estimate_NPV(self) -> float:
        """
        Estimate the Net Present Value

        Returns
        -------
        float
            the Net Present Value
        """
        return self.vpl_model.predict(self.data)[0]
    
    def estimate_MSP(self) -> float:
        """
        Estimate the Minimum Selling Price

        Returns
        -------
        float
            the Minimum Selling Price
        """
        return self.msp_model.predict(self.data)[0]
    
    def estimate_KPIs(self) -> dict:
        """
        Estimate the KPIs

        Returns
        -------
        dict
            the KPIs
        """
        return {
            'NPV': self.estimate_NPV(),
            'MSP': self.estimate_MSP()
        }
