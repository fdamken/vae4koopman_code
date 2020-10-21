import unittest

import numpy as np

from src.cholesky_update import cholesky_update


class CholeskyUpdateTest(unittest.TestCase):
    def test_basic(self):
        A = np.diag([3.0, 3.0]) ** 2
        x = np.array([4.0, 4.0])
        L = np.linalg.cholesky(A)
        A_tilde = A + np.outer(x, x)
        L_tilde = cholesky_update(L, x)

        self.assertTrue(np.allclose(A_tilde, L_tilde @ L_tilde.T))

    def test_multiple_matrices(self):
        for n in np.arange(10) + 1:
            A = np.ones((n, n)) + np.diag(np.arange(n ** 2).reshape((n, n)))
            A = (A + A.T) / 2.0
            while (np.linalg.eigvals(A) <= 0).any():
                A += np.eye(n)
            L = np.linalg.cholesky(A)
            for m in np.arange(10) + 1:
                x = np.linspace(0.0, 1.0, n) * m

                A_tilde = A + np.outer(x, x)
                L_tilde = cholesky_update(L, x)

                self.assertTrue(np.allclose(np.linalg.cholesky(A_tilde), L_tilde), 'Computed Cholesky update does not match real decomposition!')
                self.assertTrue(np.allclose(A_tilde, L_tilde @ L_tilde.T), 'Computed Cholesky update is not really a Cholesky decomposition!')

                A = A_tilde
                L = L_tilde


if __name__ == '__main__':
    unittest.main()
