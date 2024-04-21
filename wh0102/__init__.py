from typing import Self
from .descriptive_analysis.categorical_data import categorical_data
from .descriptive_analysis.continous_data import continous_data
from .regression.regression import regression

class MPHD:
    def __init__(self) -> None:
        self.continous_data = continous_data(self)
        self.categorical_data = categorical_data(self)
        self.regression = regression(self)