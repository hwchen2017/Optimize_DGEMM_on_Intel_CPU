#include <cstdio>
#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]


void dgemm_boundary_k3(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC )
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

void dgemm_kernel_v3(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{

	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}

	int m1 = M - M%2; 
	int n1 = N - N%2; 


	for(int i=0;i<m1;i+=2)
		for(int j=0;j<n1;j+=2)
		{
			double c00 = C(i, j), c01 = C(i, j+1); 
			double c10 = C(i+1, j), c11 = C(i+1, j+1); 

			for(int k=0;k<K;k++)
			{
				double ai = alpha * A(i, k), ai1 = alpha * A(i+1, k); 
				double bj = B(k, j), bj1 = B(k, j+1); 

				c00 += ai * bj; 
				c01 += ai * bj1; 
				c10 += ai1 * bj; 
				c11 += ai1 * bj1; 
			}

			C(i, j) = c00; 
			C(i, j+1) = c01; 
			C(i+1, j) = c10; 
			C(i+1, j+1) = c11; 
		}
	
	
	if(M != m1)
		dgemm_boundary_k3(M-m1, N, K, alpha, A+m1, LDA, B, LDB, C+m1, LDC); 

	if(N != n1)
		dgemm_boundary_k3(m1, N-n1, K, alpha, A, LDA, &B(0, n1), LDB, &C(0, n1), LDC); 

}