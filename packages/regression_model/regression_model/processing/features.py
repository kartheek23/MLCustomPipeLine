import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from regression_model.processing.errors import InvalidModelInputError

class LogTransformer(BaseEstimator,TransformerMixin):
	def __init__(self,variables=None):
		if not isinstance(variables,list):
			self.variables = [variables]
		else:
			self.variables = variables

	def fit(self,X,y=None):
		return self

	def transform(self,X):
		X=X.copy()
		#Check to make sure none of the values are non-negative
		if not (X[self.variables] > 0).all().all():
			vars_ = self.variables[(X[self.variables]<=0).any()]
			raise InvalidModelInputError(
				f'Variables Contain Non Negative or Zero values,'
				f"can't apply log for vars:{vars_}")
		for feature in self.variables:
			X[feature] = np.log(X[feature])

		return X