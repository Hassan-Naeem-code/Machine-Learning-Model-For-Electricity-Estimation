import unittest
import sys
from pathlib import Path
import numpy as np
# ensure project root on path so `src` package is importable when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import smape


class TestSMAPE(unittest.TestCase):
    def test_both_zero(self):
        y_true = np.array([[0.0, 0.0], [0.0, 0.0]])
        y_pred = np.array([[0.0, 0.0], [0.0, 0.0]])
        # Should treat zero/zero as zero error
        self.assertAlmostEqual(smape(y_true, y_pred), 0.0)

    def test_simple_case(self):
        y_true = np.array([[10.0, 0.0]])
        y_pred = np.array([[5.0, 0.0]])
        # manual smape: for first element: |5| / ((10+5)/2)=5/7.5=0.666666..., second element treated as 0
        expected = (5.0/7.5) * 100.0 / 2.0  # averaged over two elements
        self.assertAlmostEqual(smape(y_true, y_pred), expected)


if __name__ == '__main__':
    unittest.main()
