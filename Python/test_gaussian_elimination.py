"""
Unit tests for Gaussian Elimination Solver
"""

import unittest
from gaussian_elimination import gaussian_elimination


class TestGaussianElimination(unittest.TestCase):
    """Test cases for the gaussian_elimination function"""
    
    def test_2x2_system(self):
        """Test basic 2x2 system"""
        # System: 2x + 8y = 45
        #         5x + 5y = 345
        # Solution: 2x + 8y = 45 and 5x + 5y = 345
        # From second equation: x + y = 69, so x = 69 - y
        # Substitute: 2(69-y) + 8y = 45 → 138 - 2y + 8y = 45 → 6y = -93 → y = -15.5
        # x = 69 - (-15.5) = 84.5
        L1 = [[2, 8], [5, 5]]
        L2 = [45, 345]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 84.5, places=5)
        self.assertAlmostEqual(result[1], -15.5, places=5)
    
    def test_3x3_system(self):
        """Test 3x3 system"""
        L1 = [
            [3, 2, -1],
            [2, -2, 4],
            [-1, 0.5, -1]
        ]
        L2 = [1, -2, 0]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[1], -2.0, places=5)
        self.assertAlmostEqual(result[2], -2.0, places=5)
    
    def test_identity_matrix(self):
        """Test system with identity matrix"""
        L1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        L2 = [5, 10, 15]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        self.assertEqual(result, [5.0, 10.0, 15.0])
    
    def test_diagonal_matrix(self):
        """Test system with diagonal matrix"""
        L1 = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
        L2 = [8, 9, 16]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 4.0, places=5)
        self.assertAlmostEqual(result[1], 3.0, places=5)
        self.assertAlmostEqual(result[2], 4.0, places=5)
    
    def test_singular_matrix(self):
        """Test system with no unique solution (singular matrix)"""
        L1 = [[1, 2], [2, 4]]
        L2 = [3, 6]
        result = gaussian_elimination(L1, L2)
        self.assertIsNone(result)
    
    def test_large_coefficients(self):
        """Test system with large coefficients"""
        L1 = [[1000, 2000], [3000, 4000]]
        L2 = [5000, 11000]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 1.0, places=3)
        self.assertAlmostEqual(result[1], 2.0, places=3)
    
    def test_negative_coefficients(self):
        """Test system with negative coefficients"""
        # System: -2x + 3y = 1
        #          4x - 5y = -3
        # From first equation: 3y = 1 + 2x → y = (1 + 2x)/3
        # Substitute: 4x - 5(1 + 2x)/3 = -3 → 12x - 5 - 10x = -9 → 2x = -4 → x = -2
        # y = (1 + 2(-2))/3 = -3/3 = -1
        L1 = [[-2, 3], [4, -5]]
        L2 = [1, -3]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], -2.0, places=5)
        self.assertAlmostEqual(result[1], -1.0, places=5)
    
    def test_4x4_system(self):
        """Test larger 4x4 system"""
        L1 = [
            [2, 1, -1, 1],
            [-3, -1, 2, -2],
            [-2, 1, 2, 1],
            [1, 1, 1, 1]
        ]
        L2 = [8, -11, -3, 6]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        # Verify the solution works
        for i in range(4):
            total = sum(L1[i][j] * result[j] for j in range(4))
            self.assertAlmostEqual(total, L2[i], places=5)
    
    def test_floating_point_system(self):
        """Test system with floating point coefficients"""
        L1 = [[1.5, 2.5], [3.5, 4.5]]
        L2 = [4.0, 8.0]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        # Verify the solution
        for i in range(2):
            total = sum(L1[i][j] * result[j] for j in range(2))
            self.assertAlmostEqual(total, L2[i], places=5)
    
    def test_simple_2x2(self):
        """Test simple 2x2 system with integer solution"""
        # System: x + y = 5
        #         x - y = 1
        # Solution: x = 3, y = 2
        L1 = [[1, 1], [1, -1]]
        L2 = [5, 1]
        result = gaussian_elimination(L1, L2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 3.0, places=5)
        self.assertAlmostEqual(result[1], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()