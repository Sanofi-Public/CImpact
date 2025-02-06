import unittest
import pandas as pd
from cimpact.main import CausalImpactAnalysis


class TestUtils(unittest.TestCase):
    
    def test_pyro_causal_inference(self):
        model_config = {
            'model_type': 'pyro',  # Options: 'prophet', 'tensorflow', 'pyro'
            'model_args': {
                'standardize': False,
                'learning_rate': 0.01,
                'num_iterations': 1000,
                'num_samples': 1000
            }
        }

        file_path = './tests/unit_tests/comparison_data.csv' #'comparison_data.csv'
        index_col = 'DATE'  # Date column
        target_col = 'CHANGED'  # Target column

        data = pd.read_csv(file_path, index_col=index_col, parse_dates=True)
        pre_period = ['2019-04-16', '2019-07-14']
        post_period = ['2019-07-15', '2019-08-01']

        analysis = CausalImpactAnalysis(data,
                pre_period,
                post_period,
                model_config,
                index_col,
                target_col
                )
        result, plt = analysis.run_analysis()
         # Check that key parts of the summary string are present
        self.assertIn("Posterior inference", result)
        self.assertIn("95% CI", result)
        self.assertIn("Posterior tail-area probability p:", result)
        
    def test_tf_causal_inference(self):
        model_config = {
            'model_type': 'tensorflow',  # Options: 'prophet', 'tensorflow', 'pyro'
            'model_args': {
                'standardize': False,
                'learning_rate': 0.01,
                'num_iterations': 1000,
                'num_samples': 1000
            }
        }

        file_path = './tests/unit_tests/comparison_data.csv' #'comparison_data.csv'
        index_col = 'DATE'  # Date column
        target_col = 'CHANGED'  # Target column

        data = pd.read_csv(file_path, index_col=index_col, parse_dates=True)
        pre_period = ['2019-04-16', '2019-07-14']
        post_period = ['2019-07-15', '2019-08-01']

        analysis = CausalImpactAnalysis(data,
                pre_period,
                post_period,
                model_config,
                index_col,
                target_col
                )
        result, plt = analysis.run_analysis()
         # Check that key parts of the summary string are present
        self.assertIn("Posterior inference", result)
        self.assertIn("95% CI", result)
        self.assertIn("Posterior tail-area probability p:", result)
        
    def test_prophet_causal_inference(self):
        model_config = {
            'model_type': 'prophet',  # Options: 'prophet', 'tensorflow', 'pyro'
            'model_args': {
                'standardize': False,
                'learning_rate': 0.01,
                'num_iterations': 1000,
                'num_samples': 1000
            }
        }

        file_path = './tests/unit_tests/comparison_data.csv' #'comparison_data.csv'
        index_col = 'DATE'  # Date column
        target_col = 'CHANGED'  # Target column

        data = pd.read_csv(file_path, index_col=index_col, parse_dates=True)
        pre_period = ['2019-04-16', '2019-07-14']
        post_period = ['2019-07-15', '2019-08-01']

        analysis = CausalImpactAnalysis(data,
                pre_period,
                post_period,
                model_config,
                index_col,
                target_col
                )
        result, plt = analysis.run_analysis()
         # Check that key parts of the summary string are present
        self.assertIn("Posterior inference", result)
        self.assertIn("95% CI", result)
        self.assertIn("Posterior tail-area probability p:", result)


if __name__ == '__main__':
    test = TestUtils()
    test.test_prophet_causal_inference()
