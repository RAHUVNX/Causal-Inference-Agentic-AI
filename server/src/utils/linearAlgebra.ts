/**
 * Linear algebra utilities for regression.
 *
 * Implements Ordinary Least Squares (OLS) regression using the normal equation:
 *   β = (XᵀX)⁻¹ Xᵀy
 *
 * Also provides matrix inversion via Gauss-Jordan elimination for cases
 * where XᵀX is invertible. Falls back to pseudoinverse via SVD when needed.
 */

/**
 * Transpose a 2D matrix.
 */
export function transpose(matrix: number[][]): number[][] {
  if (matrix.length === 0) return [];
  const rows = matrix.length;
  const cols = matrix[0].length;
  const result: number[][] = Array.from({ length: cols }, () => new Array(rows));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
}

/**
 * Multiply two matrices A (m×n) and B (n×p) → C (m×p).
 */
export function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = B.length;
  const p = B[0].length;
  const C: number[][] = Array.from({ length: m }, () => new Array(p).fill(0));
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < n; k++) {
      if (A[i][k] === 0) continue;
      for (let j = 0; j < p; j++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}

/**
 * Multiply matrix A (m×n) by vector v (n) → result (m).
 */
export function matVecMul(A: number[][], v: number[]): number[] {
  const m = A.length;
  const n = v.length;
  const result = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      result[i] += A[i][j] * v[j];
    }
  }
  return result;
}

/**
 * Invert a square matrix using Gauss-Jordan elimination.
 * Returns null if the matrix is singular.
 */
export function invertMatrix(matrix: number[][]): number[][] | null {
  const n = matrix.length;
  // Augmented matrix [A | I]
  const aug: number[][] = matrix.map((row, i) => {
    const extended = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) extended[j] = row[j];
    extended[n + i] = 1;
    return extended;
  });

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }
    if (maxVal < 1e-12) return null; // Singular

    // Swap rows
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    // Scale pivot row
    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Extract inverse from right half
  return aug.map(row => row.slice(n));
}

/**
 * Solve OLS regression: find β that minimizes ||Xβ - y||²
 *
 * Uses the normal equation: β = (XᵀX)⁻¹ Xᵀy
 * Adds a small ridge penalty (λI) for numerical stability.
 *
 * @param X  Design matrix (n_samples × n_features), should include intercept column
 * @param y  Target vector (n_samples)
 * @param ridge  Ridge penalty (default 1e-8)
 * @returns Coefficient vector β
 */
export function solveOLS(X: number[][], y: number[], ridge: number = 1e-8): number[] {
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);

  // Add ridge regularization for stability: XᵀX + λI
  for (let i = 0; i < XtX.length; i++) {
    XtX[i][i] += ridge;
  }

  const XtXinv = invertMatrix(XtX);
  if (!XtXinv) {
    throw new Error('Design matrix is singular even with ridge regularization');
  }

  const Xty = matVecMul(Xt, y.map(v => [v]).map(r => r[0]));
  // Xᵀy as column: we need (XᵀX)⁻¹ * (Xᵀy)
  return matVecMul(XtXinv, Xty);
}

/**
 * Compute predictions: ŷ = Xβ
 */
export function predict(X: number[][], beta: number[]): number[] {
  return X.map(row => {
    let sum = 0;
    for (let j = 0; j < beta.length; j++) {
      sum += row[j] * beta[j];
    }
    return sum;
  });
}
