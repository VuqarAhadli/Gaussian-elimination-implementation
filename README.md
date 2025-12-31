# Gaussian Elimination Solver

A Python implementation of the Gaussian elimination algorithm with partial pivoting for solving systems of linear equations.

## Features

- Solves systems of linear equations (Ax = b)
- Implements partial pivoting for numerical stability
- Handles various matrix sizes (2x2, 3x3, 4x4, and larger)
- Detects singular matrices (no unique solution)
- Comprehensive test suite with multiple test cases

## Installation

Clone the repository:

```bash
https://github.com/VuqarAhadli/Gaussian-elimination-implementation.git
cd Gaussian-elimination-implementation
```

No additional dependencies are required beyond Python 3.x standard library.

## Usage

### Basic Example

```python
from gaussian_elimination import gaussian_elimination

# Define coefficient matrix (A)
L1 = [
    [2, 8],
    [5, 5]
]

# Define constants vector (b)
L2 = [20, 31.25]

# Solve the system
solution = gaussian_elimination(L1, L2)
# Output: [5.0, 1.25]
```

This solves the system:
```
2x + 8y = 20
5x + 5y = 31.25
```

### 3x3 System Example

```python
L1 = [
    [3, 2, -1],
    [2, -2, 4],
    [-1, 0.5, -1]
]
L2 = [1, -2, 0]

solution = gaussian_elimination(L1, L2)
# Output: [1.0, -2.0, -2.0]
```

## Algorithm

The implementation uses **Gaussian elimination with partial pivoting**:

1. **Partial Pivoting**: For each column, select the row with the largest absolute value as the pivot to minimize numerical errors
2. **Forward Elimination**: Transform the matrix to row echelon form (upper triangular)
3. **Back Substitution**: Solve for variables starting from the last equation

## Running Tests

Execute the test suite using Python's unittest framework:

```bash
python -m unittest test_gaussian_elimination.py
```

Or run with verbose output:

```bash
python -m unittest test_gaussian_elimination.py -v
```

### Test Coverage

The test suite includes:

-  Basic 2x2 systems
-  3x3 systems
-  4x4 systems
-  Identity matrices
-  Diagonal matrices
-  Singular matrices (no unique solution)
-  Systems with large coefficients
-  Systems with negative coefficients
-  Systems with floating-point coefficients

## Function Reference

### `gaussian_elimination(L1, L2)`

Solves a system of linear equations using Gaussian elimination.

**Parameters:**
- `L1` (list of lists): Coefficient matrix (n × n)
- `L2` (list): Constants vector (length n)

**Returns:**
- `list`: Solution vector if a unique solution exists
- `None`: If no unique solution exists (prints error message)

**Example:**
```python
L1 = [[1, 2], [3, 4]]
L2 = [5, 6]
result = gaussian_elimination(L1, L2)
```

## Error Handling

The function detects and reports when:
- The system has no unique solution (singular matrix)
- A zero pivot is encountered during elimination

Example output for singular matrix:
```
No unique solution for the given system
```

## Mathematical Background

Gaussian elimination solves the linear system **Ax = b** where:
- **A** is an n×n coefficient matrix
- **x** is the solution vector (unknowns)
- **b** is the constants vector

The algorithm transforms the augmented matrix [A|b] into row echelon form, then uses back substitution to find the solution.

## Limitations

- Requires a square coefficient matrix (n equations, n unknowns)
- May accumulate floating-point errors for very large systems
- Does not handle systems with infinite solutions explicitly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- Based on the classical Gaussian elimination algorithm
- Implements partial pivoting for improved numerical stability
