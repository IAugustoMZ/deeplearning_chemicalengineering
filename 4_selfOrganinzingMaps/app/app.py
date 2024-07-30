import os
from dotenv import load_dotenv
from flask_bootstrap import Bootstrap
from flask import Flask, render_template
from main.utils.forms import PlantDesignForm
from main.utils.kpis_estimate import KPIsEstimate

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

# Create an instance of the KPIsEstimate class
kpis_estimate = KPIsEstimate()

# Define the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # Create an instance of the PlantDesignForm
    form = PlantDesignForm()
    if form.validate_on_submit():
        # Prepare the data for the KPIs estimation
        kpis_estimate.prepare_data(form)

        # Estimate the KPIs
        results = kpis_estimate.estimate_KPIs()

        # get the KPIs
        npv = results.get('NPV')
        msp = results.get('MSP')

        return render_template('index.html', form=form, npv=npv, msp=msp)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)