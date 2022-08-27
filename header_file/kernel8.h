#include <cstdio>
#include <immintrin.h>

#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define ms 512
#define ks 512


void micro_kernel_8x8(int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC)
{
    
    __m512d valpha = _mm512_set1_pd(alpha); 
    __m512d a, b0, b1, b2, b3, b4, b5, b6, b7;
    
    __m512d c0 = _mm512_setzero_pd(); 
    __m512d c1 = _mm512_setzero_pd(); 
    __m512d c2 = _mm512_setzero_pd(); 
    __m512d c3 = _mm512_setzero_pd(); 
    __m512d c4 = _mm512_setzero_pd(); 
    __m512d c5 = _mm512_setzero_pd(); 
    __m512d c6 = _mm512_setzero_pd(); 
    __m512d c7 = _mm512_setzero_pd(); 

    for(int k=0;k<K;k++)
    {

        a = _mm512_mul_pd(valpha, _mm512_loadu_pd(&A[0]));
        A += 8; 

        b0 = _mm512_set1_pd(B[0]); 
        b1 = _mm512_set1_pd(B[1]); 
        b2 = _mm512_set1_pd(B[2]); 
        b3 = _mm512_set1_pd(B[3]);
        b4 = _mm512_set1_pd(B[4]); 
        b5 = _mm512_set1_pd(B[5]); 
        b6 = _mm512_set1_pd(B[6]); 
        b7 = _mm512_set1_pd(B[7]);
        
        B += 8;

        c0 = _mm512_fmadd_pd(a, b0, c0); 
        c1 = _mm512_fmadd_pd(a, b1, c1); 
        c2 = _mm512_fmadd_pd(a, b2, c2); 
        c3 = _mm512_fmadd_pd(a, b3, c3);
        c4 = _mm512_fmadd_pd(a, b4, c4); 
        c5 = _mm512_fmadd_pd(a, b5, c5); 
        c6 = _mm512_fmadd_pd(a, b6, c6); 
        c7 = _mm512_fmadd_pd(a, b7, c7);  

    }

    _mm512_storeu_pd(&C(0, 0), _mm512_add_pd(c0, _mm512_loadu_pd(&C(0, 0)))); 
    _mm512_storeu_pd(&C(0, 1), _mm512_add_pd(c1, _mm512_loadu_pd(&C(0, 1)))); 
    _mm512_storeu_pd(&C(0, 2), _mm512_add_pd(c2, _mm512_loadu_pd(&C(0, 2)))); 
    _mm512_storeu_pd(&C(0, 3), _mm512_add_pd(c3, _mm512_loadu_pd(&C(0, 3)))); 

    _mm512_storeu_pd(&C(0, 4), _mm512_add_pd(c4, _mm512_loadu_pd(&C(0, 4)))); 
    _mm512_storeu_pd(&C(0, 5), _mm512_add_pd(c5, _mm512_loadu_pd(&C(0, 5)))); 
    _mm512_storeu_pd(&C(0, 6), _mm512_add_pd(c6, _mm512_loadu_pd(&C(0, 6)))); 
    _mm512_storeu_pd(&C(0, 7), _mm512_add_pd(c7, _mm512_loadu_pd(&C(0, 7))));   
}


void pack_matrix_a8(int K, double *A, int LDA, double *Abuffer)
{
    const int bs = 8; 
    double *pt; 
    for(int j=0;j<K;j++)
    {
        pt = &A(0, j);

        for(int i=0;i<bs;i++)
        {
            *Abuffer = *(pt+i);
            Abuffer ++;
        } 
        // *Abuffer = *pt; Abuffer++; 
        // *Abuffer = *(pt+1); Abuffer++; 
        // *Abuffer = *(pt+2); Abuffer++; 
        // *Abuffer = *(pt+3); Abuffer++; 
        // *Abuffer = *(pt+4); Abuffer++; 
        // *Abuffer = *(pt+5); Abuffer++; 
        // *Abuffer = *(pt+6); Abuffer++; 
        // *Abuffer = *(pt+7); Abuffer++; 
    }
    
}
void pack_matrix_b8(int K, double *B, int LDB, double *Bbuffer)
{

    double *pt0, *pt1, *pt2, *pt3; 
    pt0 = &B(0, 0), pt1 = &B(0, 1); 
    pt2 = &B(0, 2), pt3 = &B(0, 3); 

    double *pt4, *pt5, *pt6, *pt7; 
    pt4 = &B(0, 4), pt5 = &B(0, 5); 
    pt6 = &B(0, 6), pt7 = &B(0, 7); 

    
    for(int j=0;j<K;j++)
    {

        *Bbuffer = *pt0; pt0++;Bbuffer++; 
        *Bbuffer = *pt1; pt1++;Bbuffer++; 
        *Bbuffer = *pt2; pt2++;Bbuffer++; 
        *Bbuffer = *pt3; pt3++;Bbuffer++; 

        *Bbuffer = *pt4; pt4++;Bbuffer++; 
        *Bbuffer = *pt5; pt5++;Bbuffer++; 
        *Bbuffer = *pt6; pt6++;Bbuffer++; 
        *Bbuffer = *pt7; pt7++;Bbuffer++; 

    }
    
}




void inner_kernel_v8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB,  double *C, int LDC)
{
    
    double *Abuffer, *Bbuffer; 
    Abuffer = (double *)malloc(sizeof(double)*M*K);
    Bbuffer = (double *)malloc(sizeof(double)*K*8);
    
    
    for(int j=0;j<N;j+=8)
    {
        
        pack_matrix_b8(K, &B(0, j), LDB, Bbuffer); 
        
        for(int i=0;i<M;i+=8)
		{
            
            if(j == 0) pack_matrix_a8(K, &A(i, 0), LDA, &Abuffer[i*K]); 

            
			micro_kernel_8x8(K, alpha, &Abuffer[i*K], 8, Bbuffer, LDB, &C(i, j), LDC);

		}
    }
        
    free(Bbuffer);
    free(Abuffer); 

}


void dgemm_kernel_v8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{


	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}
    
    // printf("Working kernel T3!\n");

	int mstep, kstep; 
		
    for(int kpos = 0; kpos < K; kpos += ks)
    {

        kstep = min(ks, K - kpos);

        // printf("%d \n", kpos);

        for(int mpos = 0; mpos < M; mpos += ms)
        {
            mstep = min(ms, M - mpos);

            // printf("%d \n", mpos);

            inner_kernel_v8(mstep, N, kstep, alpha, &A(mpos, kpos), LDA, &B(kpos, 0), LDB, &C(mpos, 0), LDC); 

        }	
    }

}