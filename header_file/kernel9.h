#include <cstdio>
#include <immintrin.h>

#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define min( i, j ) ( (i)<(j) ? (i): (j) )

#define ms 512
#define ks 512


void micro_kernel_16x8(int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC)
{
    
    __m512d valpha = _mm512_set1_pd(alpha); 
    __m512d a1, a2, b0, b1, b2, b3, b4, b5, b6, b7;


    __m512d c00 = _mm512_setzero_pd(); 
    __m512d c01 = _mm512_setzero_pd(); 
    __m512d c02 = _mm512_setzero_pd(); 
    __m512d c03 = _mm512_setzero_pd(); 
    __m512d c04 = _mm512_setzero_pd(); 
    __m512d c05 = _mm512_setzero_pd(); 
    __m512d c06 = _mm512_setzero_pd(); 
    __m512d c07 = _mm512_setzero_pd(); 


    __m512d c10 = _mm512_setzero_pd(); 
    __m512d c11 = _mm512_setzero_pd(); 
    __m512d c12 = _mm512_setzero_pd(); 
    __m512d c13 = _mm512_setzero_pd(); 
    __m512d c14 = _mm512_setzero_pd(); 
    __m512d c15 = _mm512_setzero_pd(); 
    __m512d c16 = _mm512_setzero_pd(); 
    __m512d c17 = _mm512_setzero_pd(); 

    for(int k=0;k<K;k++)
    {

        a1 = _mm512_mul_pd(valpha, _mm512_loadu_pd(&A[0]));
        a2 = _mm512_mul_pd(valpha, _mm512_loadu_pd(&A[8]));
        A += 16; 


        b0 = _mm512_set1_pd(B[0]); 
        b1 = _mm512_set1_pd(B[1]); 
        b2 = _mm512_set1_pd(B[2]); 
        b3 = _mm512_set1_pd(B[3]);
        b4 = _mm512_set1_pd(B[4]); 
        b5 = _mm512_set1_pd(B[5]); 
        b6 = _mm512_set1_pd(B[6]); 
        b7 = _mm512_set1_pd(B[7]);
        
        B += 8;


        c00 = _mm512_fmadd_pd(a1, b0, c00); 
        c01 = _mm512_fmadd_pd(a1, b1, c01); 
        c02 = _mm512_fmadd_pd(a1, b2, c02); 
        c03 = _mm512_fmadd_pd(a1, b3, c03);
        c04 = _mm512_fmadd_pd(a1, b4, c04); 
        c05 = _mm512_fmadd_pd(a1, b5, c05); 
        c06 = _mm512_fmadd_pd(a1, b6, c06); 
        c07 = _mm512_fmadd_pd(a1, b7, c07);

        c10 = _mm512_fmadd_pd(a2, b0, c10); 
        c11 = _mm512_fmadd_pd(a2, b1, c11); 
        c12 = _mm512_fmadd_pd(a2, b2, c12); 
        c13 = _mm512_fmadd_pd(a2, b3, c13);
        c14 = _mm512_fmadd_pd(a2, b4, c14); 
        c15 = _mm512_fmadd_pd(a2, b5, c15); 
        c16 = _mm512_fmadd_pd(a2, b6, c16); 
        c17 = _mm512_fmadd_pd(a2, b7, c17); 


    }


    _mm512_storeu_pd(&C(0, 0), _mm512_add_pd(c00, _mm512_loadu_pd(&C(0, 0)))); 
    _mm512_storeu_pd(&C(0, 1), _mm512_add_pd(c01, _mm512_loadu_pd(&C(0, 1)))); 
    _mm512_storeu_pd(&C(0, 2), _mm512_add_pd(c02, _mm512_loadu_pd(&C(0, 2)))); 
    _mm512_storeu_pd(&C(0, 3), _mm512_add_pd(c03, _mm512_loadu_pd(&C(0, 3)))); 
    _mm512_storeu_pd(&C(0, 4), _mm512_add_pd(c04, _mm512_loadu_pd(&C(0, 4)))); 
    _mm512_storeu_pd(&C(0, 5), _mm512_add_pd(c05, _mm512_loadu_pd(&C(0, 5)))); 
    _mm512_storeu_pd(&C(0, 6), _mm512_add_pd(c06, _mm512_loadu_pd(&C(0, 6)))); 
    _mm512_storeu_pd(&C(0, 7), _mm512_add_pd(c07, _mm512_loadu_pd(&C(0, 7)))); 

    _mm512_storeu_pd(&C(8, 0), _mm512_add_pd(c10, _mm512_loadu_pd(&C(8, 0)))); 
    _mm512_storeu_pd(&C(8, 1), _mm512_add_pd(c11, _mm512_loadu_pd(&C(8, 1)))); 
    _mm512_storeu_pd(&C(8, 2), _mm512_add_pd(c12, _mm512_loadu_pd(&C(8, 2)))); 
    _mm512_storeu_pd(&C(8, 3), _mm512_add_pd(c13, _mm512_loadu_pd(&C(8, 3)))); 
    _mm512_storeu_pd(&C(8, 4), _mm512_add_pd(c14, _mm512_loadu_pd(&C(8, 4)))); 
    _mm512_storeu_pd(&C(8, 5), _mm512_add_pd(c15, _mm512_loadu_pd(&C(8, 5)))); 
    _mm512_storeu_pd(&C(8, 6), _mm512_add_pd(c16, _mm512_loadu_pd(&C(8, 6)))); 
    _mm512_storeu_pd(&C(8, 7), _mm512_add_pd(c17, _mm512_loadu_pd(&C(8, 7))));  
  
}


void pack_matrix_a16x8(int K, double *A, int LDA, double *Abuffer)
{
    const int bsx = 16; 
    double *pt; 
    for(int j=0;j<K;j++)
    {
        pt = &A(0, j);

        for(int i=0;i<bsx;i++)
        {
            *Abuffer = *(pt+i);
            Abuffer ++;
        } 
    }
    
}
void pack_matrix_b16x8(int K, double *B, int LDB, double *Bbuffer)
{

    const int bsy = 8; 
    double *pt[bsy]; 

    for(int i=0;i<bsy;i++)
        pt[i] = &B(0, i); 

    
    for(int j=0;j<K;j++)
    {

        for(int i=0;i<bsy;i++)
        {
            *Bbuffer = *pt[i]; 
            pt[i]++; 
            Bbuffer ++; 
        }

    }
    
}



void inner_kernel_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB,  double *C, int LDC)
{
    
    double *Abuffer, *Bbuffer; 
    Abuffer = (double *)malloc(sizeof(double)*M*K);
    Bbuffer = (double *)malloc(sizeof(double)*K*8);
    
    
    for(int j=0;j<N;j+=8)
    {
        
        pack_matrix_b16x8(K, &B(0, j), LDB, Bbuffer); 
        
        for(int i=0;i<M;i+=16)
		{
            
            if(j == 0) pack_matrix_a16x8(K, &A(i, 0), LDA, &Abuffer[i*K]); 
 
			micro_kernel_16x8(K, alpha, &Abuffer[i*K], 16, Bbuffer, LDB, &C(i, j), LDC);

		}
    }
        
    free(Bbuffer);
    free(Abuffer); 

}


void dgemm_kernel_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{


	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}
    
    // printf("Working kernel T4!\n");

	int mstep, kstep; 
		
    for(int kpos = 0; kpos < K; kpos += ks)
    {

        kstep = min(ks, K - kpos);

        // printf("%d \n", kpos);

        for(int mpos = 0; mpos < M; mpos += ms)
        {
            mstep = min(ms, M - mpos);

            // printf("%d \n", mpos);

            inner_kernel_v9(mstep, N, kstep, alpha, &A(mpos, kpos), LDA, &B(kpos, 0), LDB, &C(mpos, 0), LDC); 

        }	
    }

}