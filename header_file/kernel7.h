#include <cstdio>
#include <immintrin.h>

#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define Kernel_k1_8x4_avx2\
	a0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i, k)));\
	a1 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i+4, k)));\
	b0 = _mm256_broadcast_sd(&B(k, j)); \
	b1 = _mm256_broadcast_sd(&B(k, j+1)); \
	b2 = _mm256_broadcast_sd(&B(k, j+2));\
	b3 = _mm256_broadcast_sd(&B(k, j+3)); \
	c00 = _mm256_fmadd_pd(a0, b0, c00); \
	c01 = _mm256_fmadd_pd(a0, b1, c01); \
	c02 = _mm256_fmadd_pd(a0, b2, c02); \
	c03 = _mm256_fmadd_pd(a0, b3, c03); \
	c10 = _mm256_fmadd_pd(a1, b0, c10); \
	c11 = _mm256_fmadd_pd(a1, b1, c11); \
	c12 = _mm256_fmadd_pd(a1, b2, c12); \
	c13 = _mm256_fmadd_pd(a1, b3, c13); \
	k++;




void dgemm_boundary_v7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC )
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

void dgemm_kernel_v7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{

	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}

	
	int m1 = M - M%8; 
	int n1 = N - N%4; 
	int k1 = K - K%4; 

	__m256d valpha = _mm256_set1_pd(alpha); 
	__m256d a0, a1,  b0, b1, b2, b3; 


	for(int i=0;i<m1;i+=8)
		for(int j=0;j<n1;j+=4)
		{
			
			__m256d c00 = _mm256_setzero_pd(); 
			__m256d c01 = _mm256_setzero_pd(); 
			__m256d c02 = _mm256_setzero_pd(); 
			__m256d c03 = _mm256_setzero_pd(); 
			__m256d c10 = _mm256_setzero_pd(); 
			__m256d c11 = _mm256_setzero_pd(); 
			__m256d c12 = _mm256_setzero_pd(); 
			__m256d c13 = _mm256_setzero_pd(); 


			for(int k=0;k<k1;)
			{
				Kernel_k1_8x4_avx2 
				Kernel_k1_8x4_avx2 
				Kernel_k1_8x4_avx2 
				Kernel_k1_8x4_avx2 
			}

			for(int k=k1;k<K;)
			{
				Kernel_k1_8x4_avx2
			}

			_mm256_storeu_pd(&C(i, j), _mm256_add_pd(c00, _mm256_loadu_pd(&C(i, j)))); 
			_mm256_storeu_pd(&C(i, j+1), _mm256_add_pd(c01, _mm256_loadu_pd(&C(i, j+1)))); 
			_mm256_storeu_pd(&C(i, j+2), _mm256_add_pd(c02, _mm256_loadu_pd(&C(i, j+2)))); 
			_mm256_storeu_pd(&C(i, j+3), _mm256_add_pd(c03, _mm256_loadu_pd(&C(i, j+3)))); 

			_mm256_storeu_pd(&C(i+4, j), _mm256_add_pd(c10, _mm256_loadu_pd(&C(i+4, j)))); 
			_mm256_storeu_pd(&C(i+4, j+1), _mm256_add_pd(c11, _mm256_loadu_pd(&C(i+4, j+1)))); 
			_mm256_storeu_pd(&C(i+4, j+2), _mm256_add_pd(c12, _mm256_loadu_pd(&C(i+4, j+2)))); 
			_mm256_storeu_pd(&C(i+4, j+3), _mm256_add_pd(c13, _mm256_loadu_pd(&C(i+4, j+3)))); 
		}
	
	
	if(M != m1)
	{
		dgemm_boundary_v7(M-m1, N, K, alpha, A+m1, LDA, B, LDB, C+m1, LDC); 
	}

	if(N != n1)
	{
		// printf("h1\n");
		dgemm_boundary_v7(m1, N-n1, K, alpha, A, LDA, &B(0, n1), LDB, &C(0, n1), LDC); 
	}

}
