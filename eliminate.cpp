#ifdef __APPLE__
  extern "C" {
    #include "vecLib/clapack.h"
    #include "vecLib/cblas.h"
  }
#else
  #include <cblas.h>
  #include <lapacke.h>
#endif

/*

  Do the partial elimination in equation system by calculating Schur complement.
  
  IMPORTANT: input matrix is columnwise ordered (Fortran-like) not in C (row-wise) fashion.

  +---+---+        +---+             +----+---+        +---+
  | A | B |}m      | E |}n           |Diag| B'|}m      | E'|}n
  +---+---+  * x = +---+       =>    +----+---+  * x = +---+
  | C | D |}k      | F |}k           | 0  | D'|}k      | F'|}k
  +---+---+        +---+             +----+---+        +---+
    ^   ^                             ^   ^
    m   k                             m   k

*/

void eliminate(double *matrix, double *rhs, int n, int m) {
  int info;
  int ipiv[m];
  int k = n - m;
  char LapackNoTrans = 'N';

  double *A = matrix, 
         *B = matrix + m * n, 
         *C = matrix + m,
         *D = matrix + n * m + m, 
         *E = rhs,
         *F = rhs + m;

  // 1: LU factorize A
  dgetrf_(
      /* M    */ &m, // size
      /* N    */ &m,
      /* A    */ A,    // pointer to data
      /* LDA  */ &n,   // LDA = matrix_size
      /* IPIV */ ipiv, // pivot vector
      /* INFO */ &info);

  if (info != 0) {
    throw lapack_exception("DGETRF failed: LU decomposition not possible");
  }

  // 2: B = A^-1 * B
  dgetrs_(
      /* TRANS */ &LapackNoTrans,
      /* N     */ &m,
      /* NRHS  */ &k,
      /* A     */ A,
      /* LDA   */ &n,
      /* IPIV  */ ipiv,
      /* B     */ B,
      /* LDB   */ &n,
      /* INFO  */ &info);

  if (info != 0) {
    throw lapack_exception("DGETRS failed");
  }

  // 3: E = A^-1 * E
  int one = 1;
  dgetrs_(
      /* TRANS */ &LapackNoTrans,
      /* N     */ &m,
      /* NRHS  */ &one,
      /* A     */ A,
      /* LDA   */ &n,
      /* IPIV  */ ipiv,
      /* B     */ E,
      /* LDB   */ &n,
      /* INFO  */ &info);

  if (info != 0) {
    throw lapack_exception("DGETRS failed");
  }

  // 4: D = D - C * B
  cblas_dgemm(CblasColMajor,
              /* TRANSA */ CblasNoTrans,
              /* TRANSB */ CblasNoTrans,
              /* M      */ k,
              /* N      */ k,
              /* K      */ m,
              /* ALPHA  */ -1.0,
              /* A      */ C,
              /* LDA    */ n,
              /* B      */ B,
              /* LDB    */ n,
              /* BETA   */ 1.0,
              /* C      */ D,
              /* LDC    */ n);

  // 5: F = F - C * E
  cblas_dgemv(CblasColMajor,
              /* TRANS */ CblasNoTrans,
              /* M     */ k,
              /* N     */ m,
              /* ALPHA */ -1.0,
              /* A     */ C,
              /* LDA   */ n,
              /* X     */ E,
              /* INCX  */ 1,
              /* BETA  */ 1.0,
              /* Y     */ F,
              /* INCY  */ 1);

  // 6.1: Zero matrix A
  for (int i = 0; i < m; i++)
    for (int j = 0; j < m; j++)
      matrix[j * n + i] = 0.0;

  // 6.2: Set 1 on diagonal of A
  for (int i = 0; i < m; i++)
    matrix[i * n + i] = 1.0;

  // 7: Zero matrix C
  for (int i = m; i < n; i++)
    for (int j = 0; j < m; j++)
      matrix[j * n + i] = 0.0;
}
