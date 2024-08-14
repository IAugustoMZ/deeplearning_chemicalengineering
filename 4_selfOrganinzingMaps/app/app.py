import os
import numpy as np
from dotenv import load_dotenv
from flask_bootstrap import Bootstrap
from flask import Flask, render_template
from main.utils.forms import PlantDesignForm
from main.utils.llm_risk_analyst import LLMRiskAnalyst
from main.utils.risk_assessment_som import RiskAssessmentSOM

# get the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# load environment variables from the .env file
load_dotenv(dotenv_path)

# get configuration settings
SECRET_KEY = os.getenv('SECRET_KEY').encode()

# Create a Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Create a Bootstrap app
Bootstrap(app)

# Create an instance of the SOM risk assessment and the LLM risk analyst
risk_assess = RiskAssessmentSOM()
llm_risk_analyst = LLMRiskAnalyst()

# Define the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # Create an instance of the PlantDesignForm
    form = PlantDesignForm()
    if form.validate_on_submit():

        # some of the form fields are percentages
        # convert them to decimal
        for item in ['glucoseToEthanol', 'celluloseToGlucose', 'organosolvToCellulose']:
            form[item].data = form[item].data / 100

        # Prepare the data for the KPIs estimation and the risk assessment
        risk_assess.prepare_data(form)

        # estimate the risk of not achieving goals
        risk = risk_assess.estimate_risk()
        riskMSP = risk.get('MSP')
        riskNPV = risk.get('NPV')

        # build distribuition for the histogram
        npv_dist = risk_assess.build_distribution(risk.get('vpl_dist'), name='NPV (MM USD)')
        msp_dist = risk_assess.build_distribution(risk.get('msp_dist'), name='MSP (USD / L)')

        # build payload
        payload = dict(
            npv=round(np.mean(risk.get('vpl_dist')), 2),
            msp=round(np.mean(risk.get('msp_dist')), 2),
            riskMSP=riskMSP,
            riskNPV=riskNPV,
            npv_dist=npv_dist,
            msp_dist=msp_dist
        )

        # assess the risk
        risk_analysis = llm_risk_analyst.assess_risk(payload, risk)

        return render_template('index.html', form=form, payload=payload, risk_analysis=risk_analysis)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)