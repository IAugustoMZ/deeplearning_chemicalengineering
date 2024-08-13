from flask_wtf import FlaskForm
from wtforms import DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class PlantDesignForm(FlaskForm):
    glucoseToEthanol = DecimalField('Glucose to Ethanol Conversion (%)', validators=[DataRequired(), NumberRange(min=90, max=100)])
    celluloseToGlucose = DecimalField('Cellulose to Glucose Conversion (%)', validators=[DataRequired(), NumberRange(min=65, max=90)])
    organosolvToCellulose = DecimalField('Organosolv to Cellulose Yield (%)', validators=[DataRequired(), NumberRange(min=80, max=100)])
    capexPhase1 = DecimalField('Capex Phase 1 (MM USD)', validators=[DataRequired(), NumberRange(min=168, max=312)])
    rawMaterialCost = DecimalField('Raw Material Cost (USD/kg)', validators=[DataRequired(), NumberRange(min=56, max=83)])
    enzymeLoading = DecimalField('Enzyme Loading (g / g cellulose)', validators=[DataRequired(), NumberRange(min=0.005, max=0.020)])
    ligninSellingPrice = DecimalField('Lignin Selling Price (USD/kg)', validators=[DataRequired(), NumberRange(min=529, max=1096)])
    ethanolSellingPrice = DecimalField('Ethanol Selling Price (USD/kg)', validators=[DataRequired(), NumberRange(min=0.26, max=0.71)])
    submit = SubmitField('Assess Project Risk')