import unittest
import pandas as pd
from prophet import Prophet
from project.modeling import train_prophet_model, make_future_dataframe, predict

class TestModeling(unittest.TestCase):
    def setUp(self):
        # Create a small dummy dataframe for Prophet
        self.df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=10, freq='D'),
            'y': [i + 0.5 for i in range(10)]
        })

    def test_train_prophet_model(self):
        # Test normal case
        model = train_prophet_model(self.df)
        self.assertIsInstance(model, Prophet)

    def test_make_future_dataframe(self):
        # Test normal case
        model = train_prophet_model(self.df)
        future = make_future_dataframe(model, periods=5)
        self.assertEqual(len(future), len(self.df) + 5)
        self.assertIn('ds', future.columns)

    def test_predict(self):
        # Test normal case
        model = train_prophet_model(self.df)
        future = make_future_dataframe(model, periods=3)
        forecast = predict(model, future)
        self.assertIn('yhat', forecast.columns)
        self.assertEqual(len(forecast), len(self.df) + 3)

    def test_empty_dataframe(self):
        # Edge case: empty dataframe
        empty_df = pd.DataFrame({'ds': [], 'y': []})
        with self.assertRaises(ValueError):
            train_prophet_model(empty_df)

    def test_missing_columns(self):
        # Edge case: missing required columns
        bad_df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=5), 'value': range(5)})
        with self.assertRaises(Exception):
            train_prophet_model(bad_df)

    def test_negative_periods(self):
        # Edge case: negative periods for future dataframe
        model = train_prophet_model(self.df)
        with self.assertRaises(ValueError):
            make_future_dataframe(model, periods=-5)

    def test_predict_with_bad_future(self):
        # Edge case: future_df missing 'ds' column
        model = train_prophet_model(self.df)
        bad_future = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=5)})
        with self.assertRaises(Exception):
            predict(model, bad_future)

if __name__ == '__main__':
    unittest.main()
