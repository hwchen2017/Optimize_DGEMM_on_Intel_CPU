#include <cstdio>
#include <immintrin.h>

#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define ms 512
#define ks 512


void micro_kernel_4x4(int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC)
{
    
    __m256d valpha = _mm256_set1_pd(alpha); 
    __m256d a, b0, b1, b2, b3;
    
     __m256d c0 = _mm256_setzero_pd(); 
     __m256d c1 = _mm256_setzero_pd(); 
     __m256d c2 = _mm256_setzero_pd(); 
     __m256d c3 = _mm256_setzero_pd(); 

    for(int k=0;k<K;k++)
    {

        a = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A[0]));
        A += 4; 
        b0 = _mm256_broadcast_sd(&B[0]); 
        b1 = _mm256_broadcast_sd(&B[1]); 
        b2 = _mm256_broadcast_sd(&B[2]); 
        b3 = _mm256_broadcast_sd(&B[3]);
        
        B += 4;

        c0 = _mm256_fmadd_pd(a, b0, c0); 
        c1 = _mm256_fmadd_pd(a, b1, c1); 
        c2 = _mm256_fmadd_pd(a, b2, c2); 
        c3 = _mm256_fmadd_pd(a, b3, c3); 

    }

    _mm256_storeu_pd(&C(0, 0), _mm256_add_pd(c0, _mm256_loadu_pd(&C(0, 0)))); 
    _mm256_storeu_pd(&C(0, 1), _mm256_add_pd(c1, _mm256_loadu_pd(&C(0, 1)))); 
    _mm256_storeu_pd(&C(0, 2), _mm256_add_pd(c2, _mm256_loadu_pd(&C(0, 2)))); 
    _mm256_storeu_pd(&C(0, 3), _mm256_add_pd(c3, _mm256_loadu_pd(&C(0, 3)))); 
    
    
}

void pack_matrix_a(int K, double *A, int LDA, double *Abuffer)
{
    
    double *pt; 
    for(int j=0;j<K;j++)
    {
        pt = &A(0, j); 
        *Abuffer = *pt; Abuffer++; 
        *Abuffer = *(pt+1); Abuffer++; 
        *Abuffer = *(pt+2); Abuffer++; 
        *Abuffer = *(pt+3); Abuffer++; 
    }
    
}
void pack_matrix_b(int K, double *B, int LDB, double *Bbuffer)
{
    double *pt0, *pt1, *pt2, *pt3; 
    pt0 = &B(0, 0), pt1 = &B(0, 1); 
    pt2 = &B(0, 2), pt3 = &B(0, 3); 
    
    for(int j=0;j<K;j++)
    {
        *Bbuffer = *pt0; pt0++;Bbuffer++; 
        *Bbuffer = *pt1; pt1++;Bbuffer++; 
        *Bbuffer = *pt2; pt2++;Bbuffer++; 
        *Bbuffer = *pt3; pt3++;Bbuffer++; 
    }
    
}




void inner_kernel_v7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB,  double *C, int LDC)
{
    
    double *Abuffer, *Bbuffer; 
    Abuffer = (double *)malloc(sizeof(double)*M*K);
    Bbuffer = (double *)malloc(sizeof(double)*K*4);
    
    
    for(int j=0;j<N;j+=4)
    {
        
        pack_matrix_b(K, &B(0, j), LDB, Bbuffer); 
        
        for(int i=0;i<M;i+=4)
		{
            
            if(j == 0) pack_matrix_a(K, &A(i, 0), LDA, &Abuffer[i*K]); 
            
			micro_kernel_4x4(K, alpha, &Abuffer[i*K], 4, Bbuffer, LDB, &C(i, j), LDC);

		}
    }
        
    free(Bbuffer);
    free(Abuffer); 

}


void dgemm_kernel_v7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{


	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}
    
    printf("Working kernel T2!\n");

	int mstep, kstep; 
		
    for(int kpos = 0; kpos < K; kpos += ks)
    {
       kstep = min(ks, K - kpos);

        for(int mpos = 0; mpos < M; mpos += ms)
        {
           mstep = min(ms, M - mpos);

            inner_kernel_v7(mstep, N, kstep, alpha, &A(mpos, kpos), LDA, &B(kpos, 0), LDB, &C(mpos, 0), LDC); 

        }	
    }

}