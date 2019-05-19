import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import pipeline
from regression_model import config
from regression_model import __version__ as _version
import logging
_logger = logging.getlogger(__name__)
def load_dataset(*,file_name:str) -> pd.DataFrame:
	_data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
	return _data

def save_pipeline(*,pipeline_to_persist) -> None:
	"""Persist the Pipeline"""

	save_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
	save_path = config.TRAINED_MODEL_DIR / save_file_name
	remove_old_pipelines(files_to_keep=save_file_name)
	joblib.dump(pipeline_to_persist, save_path)
	_logger.info(f'saved pipeline: {save_file_name}')
	

def load_pipeline(*,file_name:str) -> pipeline:
	"""Load a persisted Pipeline"""
	file_path = config.TRAINED_MODEL_DIR / file_name
	trained_model = joblib.load(filename=file_path)
    return trained_model

def remove_old_pipeline(*,files_to_keep) -> None:
	for model_file in config.TRAINED_MODEL_DIR.iterdir():
		if model_file.name not in [files_to_keep,'__init__.py']:
			model_file.unlink()