from .descriptive_analysis.categorical_data import categorical_data
from .descriptive_analysis.continous_data import continous_data
from .regression.regression import regression
from .missing_values.missing_values import missing_values
from .inferential_analysis.inferential_analysis import inferential_analysis
from .pre_processing.pre_processing import pre_processing
from .machine_learning.machine_learning import machine_learning
from .machine_learning.analyse_ml import analyse_ml

class MPHD:
    def __init__(self) -> None:
        self.continous_data = continous_data(self)
        self.categorical_data = categorical_data(self)
        self.regression = regression(self)
        self.missing_values = missing_values(self)
        self.inferential_analysis = inferential_analysis(self)
        self.pre_processing = pre_processing(self)
        self.machine_learning = machine_learning(self)
        self.analyse_ml = analyse_ml(self)