import numpy as np
import pandas as pd
from regression_model.processing.data_management import load_pipeline
from regression_model.config import config
from regression_model import __version__ as _version
import logging
print(_version)
_logger = logging.getLogger(__name__)
pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*,input_data) -> dict:
	"""Make a prediction using the saved pipeline"""
	data = pd.read_json(input_data)
	prediction = _price_pipe.predict(data[config.FEATURES])
	output = np.exp(prediction)
	results = {'predictions':output,'version':_version}
	_logger.info(
		f'Making predictions with model version:{_version}'
		f'Inputs: {validated_data} '
        f'Predictions: {results}')
	return results
