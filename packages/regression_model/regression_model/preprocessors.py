import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class CategoricalImputer(BaseEstimator,TransformerMixin):
	""" Categorical Data Missing value Imputer """
	def __init__(self,variables = None) -> None:
		if not isinstance(variables,list):
			self.variables = [variables]
		else:
			self.variables = variables

	def fit(self,X:pd.DataFrame,y:pd.Series = None) -> 'CategoricalImputer':
		""" Fit Statement to accomodate the sklearn Pipeline"""
		return self

	def transform(self,X: pd.DataFrame) -> pd.DataFrame:
		""" Apply the transform to DataFrame"""
		X=X.copy()
		for feature in self.variables:
			X[feature] = X[feature].fillna("Missing")

		return X

class NumericalImputer(BaseEstimator,TransformerMixin):
	""" Numerical Missing Value Imputer"""
	def __init__(self,variables=None):
		if not isinstance(variables,list):
			self.variables = [variables]
		else:
			self.variables = variables

	def fit (self ,X,y=None):
		##Persist Mode in a dictionary
		self.imputer_dict = {}
		for feature in self.variables:
			self.imputer_dict[feature] = X[feature].mode()[0]
		return self

	def transform(self,X):
		X=X.copy()
		for feature in self.variables:
			X[feature].fillna(self.imputer_dict[feature],inplace=True)
		return X

class TemporalVariableEstimator(BaseEstimator,TransformerMixin):

	def __init__(self,variables=None,reference_variable=None):
		if not isinstance(variables,list):
			self.variables = [variables]
		else:
			self.variables = variables
		self.reference_variable = reference_variable

	def fit(self,X,y=None):
		return self

	def transform(self,X):
		X=X.copy()
		for feature in self.variables:
			X[feature] = X[self.reference_variable] - X[feature]

		return X

class RareLabelCategoricalEncoder(BaseEstimator,TransformerMixin):
	def __init__(self,tol=0.05,variables=None):
		self.tol = tol
		if not isinstance(variables,list):
			self.variables = [variables]
		else:
			self.variables = variables

	def fit(self,X,y=None):
		self.encoder_dict_ = {}
		for var in self.variables:
			t = pd.Series(X[var].value_counts()/np.float(len(X)))
			self.encoder_dict_[var] = list(t[t>=self.tol].index)
		return self

	def transform(self,X):
		X=X.copy()

		for var in self.variables:
			X[var] = np.where(X[var].isin(self.encoder_dict_[var]),X[var],"Rare")

		return X


class CategoricalEncoder(BaseEstimator,TransformerMixin):
	"""String to Numbers Categorical Encoder"""
	def __init__(self,variables=None):
		if not isinstance(variables,list):
			self.variables = [variables]
		else:
			self.variables = variables

	def fit(self,X,y):
		temp = pd.concat([X,y],axis=1)
		temp.columns = list(X.columns) + ['target']

		#persist transforming dictionary
		self.encoder_dict_ = {}
		for var in self.variables:
			t=temp.groupby([var])['target'].mean().sort_values(ascending=True).index
			self.encoder_dict_[var] = {k:i for i,k in enumerate(t,0)}

		return self

	def transform(self,X):
		X=X.copy()
		for var in self.variables:
			X[var] = X[var].map(self.encoder_dict_[var])

		#Check if transformer introduces NAN
		if X[self.variables].isnull().any().any():
			null_counts = X[self.variables].isnull().any()
			vars_ = {key: value for(key,value) in null_counts.item()
			if value is True}

			raise ValueError(
				f'Categorical encoder has returned NAN when'
				f'Transforming categorical variables: {vars_.keys()}')

		return X
	
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
			raise ValueError(
				f'Variables Contain Non Negative or Zero values,'
				f"can't apply log for vars:{vars_}")
		for feature in self.variables:
			X[feature] = np.log(X[feature])

		return X

class DropUnneccessaryVariables(BaseEstimator,TransformerMixin):
	def __init__(self,variables_to_drop=None):
		self.variables_to_drop = variables_to_drop

	def fit(self,X,y=None):
		return self

	def transform(self,X):
		X = X.copy()
		X = X.drop(self.variables_to_drop,axis=1)

		return X








































































































































