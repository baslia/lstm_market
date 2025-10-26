import unittest
import numpy as np
from src import predict_lstm as pl


class TestPredictLSTM(unittest.TestCase):
    def test_create_sequences(self):
        # values: 5 rows, 2 features
        values = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
        seq_len = 2
        X, y = pl.create_sequences(values, seq_len)
        # expect 3 sequences
        self.assertEqual(X.shape, (3, seq_len, 2))
        self.assertEqual(y.shape, (3, 2))
        # check first sequence and target
        np.testing.assert_array_equal(X[0], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(y[0], np.array([5, 6]))

    def test_normalize_input_shape_with_tuple(self):
        shape = (20, 2)
        out = pl._normalize_input_shape(shape)
        self.assertEqual(out, (20, 2))

    def test_normalize_input_shape_with_object_shape(self):
        class Dummy:
            shape = (None, 20, 2)

        d = Dummy()
        out = pl._normalize_input_shape(d)
        self.assertEqual(out, (20, 2))

    @unittest.skipUnless(pl.TF_AVAILABLE, "TensorFlow is not available in this environment")
    def test_build_model_from_params(self):
        # basic smoke test for the model builder
        params = {'units': 32, 'dropout': 0.1, 'dense_units': 16, 'lr': 1e-3}
        model = pl.build_model_from_params((20, 2), params)
        # ensure model produces expected output shape for a batch of inputs
        x = np.random.rand(2, 20, 2).astype('float32')
        y = model.predict(x)
        self.assertEqual(y.shape, (2, 2))

    @unittest.skipUnless(pl.TF_AVAILABLE, "TensorFlow is not available in this environment")
    def test_tune_hyperparams_smoke(self):
        # small synthetic data to exercise the tuner
        seq_len = 10
        X_train = np.random.rand(30, seq_len, 2).astype('float32')
        y_train = np.random.rand(30, 2).astype('float32')
        X_val = np.random.rand(6, seq_len, 2).astype('float32')
        y_val = np.random.rand(6, 2).astype('float32')
        best_params, best_loss = pl.tune_hyperparams(X_train, y_train, X_val, y_val, (seq_len, 2), max_trials=1)
        self.assertIsInstance(best_params, dict)
        self.assertIn('units', best_params)


if __name__ == '__main__':
    unittest.main()

