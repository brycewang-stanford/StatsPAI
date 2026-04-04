"""
Core base classes and interfaces for StatsPAI
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all econometric models
    """
    
    def __init__(self):
        self.is_fitted = False
        self._results = None
    
    @abstractmethod
    def fit(self, **kwargs) -> 'EconometricResults':  # noqa: F821
        """
        Fit the econometric model
        
        Returns
        -------
        EconometricResults
            Fitted model results
        """
        pass
    
    @abstractmethod
    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate predictions from the fitted model
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data for prediction. If None, uses training data.
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        pass
    
    def summary(self) -> str:
        """
        Return a summary of the fitted model
        
        Returns
        -------
        str
            Model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling summary()")
        return self._results.summary()


class BaseEstimator(ABC):
    """
    Abstract base class for estimation algorithms
    """
    
    @abstractmethod
    def estimate(self, y: np.ndarray, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Estimate model parameters
        
        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Independent variables
        **kwargs
            Additional estimation options
            
        Returns
        -------
        Dict[str, Any]
            Estimation results including parameters, standard errors, etc.
        """
        pass
