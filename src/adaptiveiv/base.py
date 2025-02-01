from abc import ABC, abstractmethod

class BaseIVEstimator(ABC):
    """Base class for IV estimators"""
    
    @abstractmethod
    def fit(self, y, x, z, groups):
        """Fit the model"""
        pass

    @abstractmethod
    def predict(self, x):
        """Make predictions"""
        pass

    @abstractmethod
    def get_diagnostics(self):
        """Return diagnostic information"""
        pass