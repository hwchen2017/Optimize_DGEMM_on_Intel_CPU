#define A(i, k) A[(i) + (k)*LDA]
#define B(k, j) B[(k) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]



void dgemm_kernel_v1(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC)
{

	if(beta != 1.0)
	{
		for(int i=0;i<M*N;i++)
			C[i] *= beta; 
	}

	for(int i=0;i<M;i++)
		for(int j=0;j<N;j++)
		{

			for(int k=0;k<K;k++)
				C(i, j) += alpha * A(i, k) * B(k, j);

		}

}