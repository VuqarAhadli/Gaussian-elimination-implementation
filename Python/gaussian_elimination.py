"""
    Gaussian Elimination Solver
    A Python implementation of the Gaussian elimination algorithm with partial pivoting
    for solving systems of linear equations.

    Solve a system of linear equations using Gaussian elimination with partial pivoting.
    
    Parameters:
    -----------
    L1 : list of lists
        Coefficient matrix (n x n) where n is the number of equations
    L2 : list
        Constants vector (length n)
    
    Returns:
    --------
    list or None
        Solution vector if a unique solution exists, None otherwise
    
    Examples:
    ---------
    >>> L1 = [[2, 8], [5, 5]]
    >>> L2 = [45, 345]
    >>> gaussian_elimination(L1, L2)
    [60.0, -9.375]
"""

def gaussian_elimination(L1, L2):
    n = len(L1)
    

    if len(L1[0]) == 1 and n > 1:
        L1 = [[L1[i][0] if i == j else 0 for j in range(n)] for i in range(n)]
    

    matrix = [L1[i] + [L2[i]] for i in range(n)]
    

    for i in range(n):
        row = i
        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(matrix[row][i]):
                row = j
        
        if matrix[row][i] == 0:
            print("No unique solution for the given system")
            return None
        
        matrix[i], matrix[row] = matrix[row], matrix[i]
        
        pivot = matrix[i][i]
        for k in range(i, n + 1):
            matrix[i][k] /= pivot
        
        for l in range(i + 1, n):
            factor = matrix[l][i]
            for m in range(i, n + 1):
                matrix[l][m] -= factor * matrix[i][m]
    
    result = [0] * n
    for i in range(n - 1, -1, -1):
        result[i] = matrix[i][n] - sum(matrix[i][j] * result[j] for j in range(i + 1, n))
    
    print(result)
    return result


if __name__ == "__main__":
  
    print("Example: Solving 2x2 system")
    L1 = [
        [2, 8],
        [5, 5]
    ]
    L2 = [45, 345]
    print(f"System: {L1} x = {L2}")
    gaussian_elimination(L1, L2)
