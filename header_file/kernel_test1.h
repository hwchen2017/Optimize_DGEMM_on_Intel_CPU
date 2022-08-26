#include <cstdio>
#include <immintrin.h>

#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define ms 256
#define ks 512



void inner_kernel_t1(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB,  double *C, int LDC)
{


	__m256d valpha = _mm256_set1_pd(alpha); 
    __m256d c0, c1, c2, c3; 
    __m256d a, b0, b1, b2, b3;

    for(int j=0;j<N;j+=4)
        for(int i=0;i<M;i+=4)
		{
			
			c0 = _mm256_setzero_pd(); 
			c1 = _mm256_setzero_pd(); 
			c2 = _mm256_setzero_pd(); 
			c3 = _mm256_setzero_pd(); 

			for(int k=0;k<K;k++)
			{
				
				a = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i, k))); 
				b0 = _mm256_broadcast_sd(&B(k, j)); 
				b1 = _mm256_broadcast_sd(&B(k, j+1)); 
				b2 = _mm256_broadcast_sd(&B(k, j+2)); 
				b3 = _mm256_broadcast_sd(&B(k, j+3)); 

				c0 = _mm256_fmadd_pd(a, b0, c0); 
				c1 = _mm256_fmadd_pd(a, b1, c1); 
				c2 = _mm256_fmadd_pd(a, b2, c2); 
				c3 = _mm256_fmadd_pd(a, b3, c3); 
			
			}

			_mm256_storeu_pd(&C(i, j), _mm256_add_pd(c0, _mm256_loadu_pd(&C(i, j)))); 
			_mm256_storeu_pd(&C(i, j+1), _mm256_add_pd(c1, _mm256_loadu_pd(&C(i, j+1)))); 
			_mm256_storeu_pd(&C(i, j+2), _mm256_add_pd(c2, _mm256_loadu_pd(&C(i, j+2)))); 
			_mm256_storeu_pd(&C(i, j+3), _mm256_add_pd(c3, _mm256_loadu_pd(&C(i, j+3)))); 

		}

}


void dgemm_kernel_t1(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{


	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}
    
    printf("Working kernel T1!\n");

	int mstep, kstep; 
		
    for(int kpos = 0; kpos < K; kpos += ks)
    {
       kstep = min(ks, K - kpos);

        for(int mpos = 0; mpos < M; mpos += ms)
        {
           mstep = min(ms, M - mpos);

            inner_kernel_t1(mstep, N, kstep, alpha, &A(mpos, kpos), LDA, &B(kpos, 0), LDB, &C(mpos, 0), LDC); 

        }	
    }

}