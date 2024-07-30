from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from wtforms import DecimalField, SubmitField
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, NumberRange
from wtforms import DecimalField, SubmitField

class PlantDesignForm(FlaskForm):
    glucoseToEthanol = DecimalField('Glucose to Ethanol Conversion (%)', validators=[DataRequired(), NumberRange(min=0, max=100)])
    celluloseToGlucose = DecimalField('Cellulose to Glucose Conversion (%)', validators=[DataRequired(), NumberRange(min=0, max=100)])
    organosolvToCellulose = DecimalField('Organosolv to Cellulose Yield (%)', validators=[DataRequired(), NumberRange(min=0, max=100)])
    capexPhase1 = DecimalField('Capex Phase 1 (MM USD)', validators=[DataRequired(), NumberRange(min=100, max=500)])
    rawMaterialCost = DecimalField('Raw Material Cost (USD/kg)', validators=[DataRequired(), NumberRange(min=0, max=1000)])
    enzymeLoading = DecimalField('Enzyme Loading (g / g cellulose)', validators=[DataRequired(), NumberRange(min=0, max=1000)])
    ligninSellingPrice = DecimalField('Lignin Selling Price (USD/kg)', validators=[DataRequired(), NumberRange(min=0, max=1000)])
    ethanolSellingPrice = DecimalField('Ethanol Selling Price (USD/kg)', validators=[DataRequired(), NumberRange(min=0, max=1000)])
    submit = SubmitField('Assess Project Risk')