#include <cstdio>
#include <immintrin.h>

#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define Kernel_k1_4x4_avx2\
	a = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i, k)));\
	b0 = _mm256_broadcast_sd(&B(k, j)); \
	b1 = _mm256_broadcast_sd(&B(k, j+1)); \
	b2 = _mm256_broadcast_sd(&B(k, j+2));\
	b3 = _mm256_broadcast_sd(&B(k, j+3)); \
	c0 = _mm256_fmadd_pd(a, b0, c0); \
	c1 = _mm256_fmadd_pd(a, b1, c1); \
	c2 = _mm256_fmadd_pd(a, b2, c2); \
	c3 = _mm256_fmadd_pd(a, b3, c3); \
	k++;




void dgemm_boundary_v6(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC )
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

void dgemm_kernel_v6(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{

	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}
    
	int m1 = M - M%4; 
	int n1 = N - N%4; 
	int k1 = K - K%4; 

	__m256d valpha = _mm256_set1_pd(alpha); 
	__m256d a, b0, b1, b2, b3; 


	for(int i=0;i<m1;i+=4)
		for(int j=0;j<n1;j+=4)
		{
			
			__m256d c0 = _mm256_setzero_pd(); 
			__m256d c1 = _mm256_setzero_pd(); 
			__m256d c2 = _mm256_setzero_pd(); 
			__m256d c3 = _mm256_setzero_pd(); 

			for(int k=0;k<k1;)
			{
				Kernel_k1_4x4_avx2 
				Kernel_k1_4x4_avx2 
				Kernel_k1_4x4_avx2 
				Kernel_k1_4x4_avx2 
			}

			for(int k=k1;k<K;)
			{
				Kernel_k1_4x4_avx2
			}

			_mm256_storeu_pd(&C(i, j), _mm256_add_pd(c0, _mm256_loadu_pd(&C(i, j)))); 
			_mm256_storeu_pd(&C(i, j+1), _mm256_add_pd(c1, _mm256_loadu_pd(&C(i, j+1)))); 
			_mm256_storeu_pd(&C(i, j+2), _mm256_add_pd(c2, _mm256_loadu_pd(&C(i, j+2)))); 
			_mm256_storeu_pd(&C(i, j+3), _mm256_add_pd(c3, _mm256_loadu_pd(&C(i, j+3)))); 
		}
	
	
	if(M != m1)
		dgemm_boundary_v6(M-m1, N, K, alpha, A+m1, LDA, B, LDB, C+m1, LDC);

	if(N != n1)
		dgemm_boundary_v6(m1, N-n1, K, alpha, A, LDA, &B(0, n1), LDB, &C(0, n1), LDC); 

}
