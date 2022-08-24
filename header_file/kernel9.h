#include <cstdio>
#include <immintrin.h>

#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define mb_size 192
#define nb_size 1024
#define kb_size 384
// const int mb_size = 192;
// const int nb_size = 1024;
// const int kb_size = 384;

void dgemm_boundary_k9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC )
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

#define Kernel_k9_4x4_avx2 \
	a = _mm256_mul_pd(valpha, _mm256_loadu_pd(pack_a));\
	b0 = _mm256_broadcast_sd(pack_b); \
	b1 = _mm256_broadcast_sd(pack_b+1); \
	b2 = _mm256_broadcast_sd(pack_b+2);\
	b3 = _mm256_broadcast_sd(pack_b+3); \
	c0 = _mm256_fmadd_pd(a, b0, c0); \
	c1 = _mm256_fmadd_pd(a, b1, c1); \
	c2 = _mm256_fmadd_pd(a, b2, c2); \
	c3 = _mm256_fmadd_pd(a, b3, c3); \
	pack_a += 8;\
	pack_b += 4;\
	k++;


#define kernel_4xkx4_packing \
	__m256d c0 = _mm256_setzero_pd(); \
	__m256d c1 = _mm256_setzero_pd(); \
	__m256d c2 = _mm256_setzero_pd(); \
	__m256d c3 = _mm256_setzero_pd(); \
	for(int k=0;k<k1;){\
		Kernel_k9_4x4_avx2\
		Kernel_k9_4x4_avx2 \
		Kernel_k9_4x4_avx2 \
		Kernel_k9_4x4_avx2 \
	}\
	for(int k=k1;k<K;)\
	{\
		Kernel_k9_4x4_avx2\
	}\
	_mm256_storeu_pd(&C(i, j), _mm256_add_pd(c0, _mm256_loadu_pd(&C(i, j))));\
	_mm256_storeu_pd(&C(i, j+1), _mm256_add_pd(c1, _mm256_loadu_pd(&C(i, j+1))));\
	_mm256_storeu_pd(&C(i, j+2), _mm256_add_pd(c2, _mm256_loadu_pd(&C(i, j+2)))); \
	_mm256_storeu_pd(&C(i, j+3), _mm256_add_pd(c3, _mm256_loadu_pd(&C(i, j+3))));

#define Kernel_k9_8x4_avx2 \
	a0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(pack_a));\
	a1 = _mm256_mul_pd(valpha, _mm256_loadu_pd(pack_a+4));\
	b0 = _mm256_broadcast_sd(pack_b);\
	b1 = _mm256_broadcast_sd(pack_b+1);\
	b2 = _mm256_broadcast_sd(pack_b+2);\
	b3 = _mm256_broadcast_sd(pack_b+3);\
	c00 = _mm256_fmadd_pd(a0, b0, c00);\
	c01 = _mm256_fmadd_pd(a0, b1, c01);\
	c02 = _mm256_fmadd_pd(a0, b2, c02); \
	c03 = _mm256_fmadd_pd(a0, b3, c03); \
	c10 = _mm256_fmadd_pd(a1, b0, c10); \
	c11 = _mm256_fmadd_pd(a1, b1, c11); \
	c12 = _mm256_fmadd_pd(a1, b2, c12); \
	c13 = _mm256_fmadd_pd(a1, b3, c13); \
	pack_a += 8;\
	pack_b += 4;\
	k++;

#define kernel_8xkx4_packing \
	__m256d c00 = _mm256_setzero_pd();\
	__m256d c01 = _mm256_setzero_pd();\
	__m256d c02 = _mm256_setzero_pd();\
	__m256d c03 = _mm256_setzero_pd();\
	__m256d c10 = _mm256_setzero_pd();\
	__m256d c11 = _mm256_setzero_pd();\
	__m256d c12 = _mm256_setzero_pd();\
	__m256d c13 = _mm256_setzero_pd();\
	for(int k=0;k<k1;)\
	{\
		Kernel_k1_8x4_avx2\
		Kernel_k1_8x4_avx2\
		Kernel_k1_8x4_avx2\
		Kernel_k1_8x4_avx2\
	}\
	for(int k=k1;k<K;)\
	{\
		Kernel_k1_8x4_avx2\
	}\
	_mm256_storeu_pd(&C(i, j), _mm256_add_pd(c00, _mm256_loadu_pd(&C(i, j))));\
	_mm256_storeu_pd(&C(i, j+1), _mm256_add_pd(c01, _mm256_loadu_pd(&C(i, j+1))));\
	_mm256_storeu_pd(&C(i, j+2), _mm256_add_pd(c02, _mm256_loadu_pd(&C(i, j+2))));\
	_mm256_storeu_pd(&C(i, j+3), _mm256_add_pd(c03, _mm256_loadu_pd(&C(i, j+3))));\
	_mm256_storeu_pd(&C(i+4, j), _mm256_add_pd(c10, _mm256_loadu_pd(&C(i+4, j))));\
	_mm256_storeu_pd(&C(i+4, j+1), _mm256_add_pd(c11, _mm256_loadu_pd(&C(i+4, j+1))));\
	_mm256_storeu_pd(&C(i+4, j+2), _mm256_add_pd(c12, _mm256_loadu_pd(&C(i+4, j+2))));\
	_mm256_storeu_pd(&C(i+4, j+3), _mm256_add_pd(c13, _mm256_loadu_pd(&C(i+4, j+3))));






void sub_dgemm_kernel_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double *C, int LDC)
{

	int m1 = M - M%8; 
	int n1 = N - N%4; 
	int k1 = K - K%4; 

	double *pack_a = A; 
	double *pack_b = B; 


	__m256d valpha = _mm256_set1_pd(alpha); 
	__m256d a, a0, a1, b0, b1, b2, b3; 
	__m256d c00, c01, c02, c03, c10, c11, c12, c13; 
	__m256d c0, c1, c2, c3; 

	for(int i=0;i<m1;i+=8)
		for(int j=0;j<n1;j+=4)
		{
			pack_a = A + i*K; 
			pack_b = B + j*K; 

			kernel_8xkx4_packing
		}

	for(int i=m1;i<M;i+=4)
		for(int j=n1;j<N;j+=4)
		{
			pack_a = A + i*K; 
			pack_b = B + j*K; 

			kernel_4xkx4_packing
		}
}

void pack_matrix_a(double *A, double *buffer, int LDA, int ms, int ks)
{
	double *pa, *pb; 
	pb = buffer; 

	int lm = ms, mi, ki; 

	for(mi = 0; lm>8; mi += 8, lm -= 8 )
	{
		pa = A + mi; 

		for(ki = 0; ki < ks; ki ++)
		{
			_mm512_store_pd(pb, _mm512_loadu_pd(pa)); 
			pa += LDA; 
			pb += 8; 
		}
	} 

	for(; lm>3; mi += 4, lm -= 4 )
	{
		pa = A + mi; 

		for(ki = 0; ki < ks; ki ++)
		{
			_mm256_store_pd(pb, _mm256_loadu_pd(pa)); 
			pa += LDA; 
			pb += 4; 
		}
	} 
}


void pack_matrix_b(double *B, double *buffer, int LDB, int ks, int ns )
{
	double *p1, *p2, *p3, *p4, *pb;

	pb = buffer; 

	for(int  ni = 0; ni < ns; ni += 4 )
	{
		p1 = B + ni * LDB;
		p2 = p1 + LDB; 
		p3 = p2 + LDB; 
		p4 = p3 + LDB; 

		for(int ki=0;ki<ks;ki++)
		{
			pb = p1; p1++; pb++; 
			pb = p2; p2++; pb++; 
			pb = p3; p3++; pb++; 
			pb = p4; p4++; pb++; 
		}
	}


}



void dgemm_kernel_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{


	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}

	double *b_buffer = (double*) aligned_alloc(4096, kb_size*nb_size * sizeof(double)); 
	double *a_buffer = (double*) aligned_alloc(4096, kb_size*mb_size * sizeof(double)); 
	
	int mstep, kstep, nstep; 

	for(int npos = 0; npos < N; npos += nstep)
	{
		if(N - npos >= nb_size) nstep = nb_size;
		else nstep = N - npos; 
		
		for(int kpos = 0; kpos < K; kpos += kstep)
		{
			if(K - kpos > kb_size) kstep = kb_size; 
			else kstep = K - kpos; 

			pack_matrix_b(B+kpos+npos*LDB, b_buffer, LDB, kstep, nstep);  

			for(int mpos = 0; mpos < M; mpos += mstep)
			{

				if(M - mpos > mb_size) mstep = mb_size; 
				else mstep = M - mpos; 

				pack_matrix_a(A+mpos + kpos*LDA, a_buffer, LDA, mstep, kstep); 


				sub_dgemm_kernel_v9(mstep, nstep, kstep, alpha, &A(mpos, kpos), LDA, &B(kpos, npos), LDB, &C(mpos, npos), LDC); 


			}	
		}

	}






}
