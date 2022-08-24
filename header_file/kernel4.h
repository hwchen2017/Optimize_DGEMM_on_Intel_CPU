#include <cstdio>
#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]


void dgemm_boundary_v4(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC )
{

	for(int i=0;i<M;i++)
		for(int j=0;j<N;j++)
		{
			double tmp = C(i, j); 
			for(int k=0;k<K;k++)
				tmp += alpha * A(i, k) * B(k, j); 

			C(i, j) = tmp; 
		}

}

void dgemm_kernel_v4(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{

	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}

	
	int m1 = M - M%4; 
	int n1 = N - N%4; 


	for(int i=0;i<m1;i+=4)
		for(int j=0;j<n1;j+=4)
		{
			double c00 = C(i, j), c01 = C(i, j+1); 
			double c02 = C(i, j+2), c03 = C(i, j+3); 

			double c10 = C(i+1, j), c11 = C(i+1, j+1); 
			double c12 = C(i+1, j+2), c13 = C(i+1, j+3);

			double c20 = C(i+2, j), c21 = C(i+2, j+1); 
			double c22 = C(i+2, j+2), c23 = C(i+2, j+3);

			double c30 = C(i+3, j), c31 = C(i+3, j+1); 
			double c32 = C(i+3, j+2), c33 = C(i+3, j+3);


			for(int k=0;k<K;k++)
			{
				double a0 = alpha * A(i, k), a1 = alpha * A(i+1, k);
				double a2 = alpha * A(i+2, k), a3 = alpha * A(i+3, k); 

				double b0 = B(k, j), b1 = B(k, j+1);
				double b2 = B(k, j+2), b3 = B(k, j+3);  

				c00 += a0 * b0, c01 += a0 * b1; 
				c02 += a0 * b2, c03 += a0 * b3; 

				c10 += a1 * b0, c11 += a1 * b1; 
				c12 += a1 * b2, c13 += a1 * b3; 

				c20 += a2 * b0, c21 += a2 * b1; 
				c22 += a2 * b2, c23 += a2 * b3; 

				c30 += a3 * b0, c31 += a3 * b1; 
				c32 += a3 * b2, c33 += a3 * b3; 
			
			}

			C(i, j) = c00, C(i, j+1) = c01; 
			C(i, j+2) = c02, C(i, j+3) = c03; 

			C(i+1, j) = c10, C(i+1, j+1) = c11; 
			C(i+1, j+2) = c12, C(i+1, j+3) = c13; 

			C(i+2, j) = c20, C(i+2, j+1) = c21; 
			C(i+2, j+2) = c22, C(i+2, j+3) = c23; 

			C(i+3, j) = c30, C(i+3, j+1) = c31; 
			C(i+3, j+2) = c32, C(i+3, j+3) = c33; 


		}
	
	
	if(M != m1)
	{
		dgemm_boundary_v4(M-m1, N, K, alpha, A+m1, LDA, B, LDB, C+m1, LDC); 
	}

	if(N != n1)
	{
		// printf("h1\n");
		dgemm_boundary_v4(m1, N-n1, K, alpha, A, LDA, &B(0, n1), LDB, &C(0, n1), LDC); 
	}

}