import os

# define constants
DATASOURCE = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ), '0_data', 'rnn_studies','BrentOilPrices.csv'
)
MODEL_SINK = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))
    ), '1_models', 'rnn_studies'
)