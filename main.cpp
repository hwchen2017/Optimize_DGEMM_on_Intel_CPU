#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdbool>
#include <chrono>
#include <mkl.h>
#include <unistd.h>
#include "utils.h"
#include "kernels.h"

using namespace std; 

void test_dgemm_kernel(int num, int M, int N, int K, double alpha, double *A, double *B, double beta, double *C)
{
	switch(num)
	{
		case 1: dgemm_kernel_v1(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;
		case 2: dgemm_kernel_v2(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break; 
		case 3: dgemm_kernel_v3(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;
		case 4: dgemm_kernel_v4(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;  
		case 5: dgemm_kernel_v5(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;
		case 6: dgemm_kernel_v6(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;
		case 7: dgemm_kernel_v7(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break; 
		case 8: dgemm_kernel_v8(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;
        case 9: dgemm_kernel_v9(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;
		case 0: dgemm_kernel_v9(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;       
	}
}


int main(int argc, char *argv[])
{

	int kernel_num = 0;
	int sys_size = 2048; 

	char ch; 
	while((ch = getopt(argc, argv, "k:n:")) != EOF)
	{
		switch(ch)
		{
			case 'k' : kernel_num = atoi(optarg);
			break; 
			case 'n' : sys_size = atoi(optarg); 
			break; 

		}
	}

	if(kernel_num > 9 or kernel_num < 0)
		kernel_num = 0; 
	
	int M, N, K;
	double *A, *B, *C, *C_mkl;
	double alpha = 1.0, beta = 0.0;  

	M = sys_size, N = sys_size, K = sys_size; 
	
	printf("\nMatrix Size: %d X %d, %d X %d\n", M, K, K, N ); 
	cout<<"Kernel number: "<<kernel_num<<endl<<endl;


	A = (double *)malloc(sizeof(double) * M * K);
	B = (double *)malloc(sizeof(double) * K * N); 
	C = (double *)malloc(sizeof(double) * M * N); 
	C_mkl = (double *)malloc(sizeof(double) * M *N);  


	random_initial_matrix(A, M, K); 
	random_initial_matrix(B, K, N); 
	random_initial_matrix(C, M, N); 
	random_initial_matrix(C_mkl, M, N); 



	// memset(C, 0, sizeof(C)); 
	// memset(C_mkl, 0, sizeof(C_mkl));

    memset(C, 0, sizeof(C));

    test_dgemm_kernel(kernel_num, M, N, K, alpha, A, B, beta, C); 

	auto start = std::chrono::high_resolution_clock::now();

	for(int i=0;i<10;i++)
	{
		test_dgemm_kernel(kernel_num, M, N, K, alpha, A, B, beta, C); 
	}
	

	auto end = std::chrono::high_resolution_clock::now(); 
	auto elapsed = end - start;

	double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() * 1e-3;
	time_ms /= 10.0; 

	cout<<"===================MY CPU code====================="<<endl;
	cout<<"Elapsed Time: "<<time_ms<<" ms"<<endl;
    cout<<2.*1e-6*M*N*K/time_ms<<endl<<endl; 
    
//     memset(C, 0, sizeof(C));
    
//     start = std::chrono::high_resolution_clock::now();
// 	MY_MMult(M, N, K, A, M, B, K, C, N); 

// 	end = std::chrono::high_resolution_clock::now(); 
// 	elapsed = end - start;

// 	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() * 1e-3;

// 	cout<<"===================Flame CPU code====================="<<endl;
// 	cout<<"Elapsed Time: "<<time_ms<<" ms"<<endl<<endl;


	mkl_set_num_threads(1); 

	//warm up
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C_mkl, M);


// 	cout<<"Time: "<<time_ms<<" ms"<<endl;


	start = std::chrono::high_resolution_clock::now();

	for(int i=0;i<10;i++)
	{
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C_mkl, M);
	}

	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start; 

	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() * 1e-3 ;
	time_ms /= 10.0;

	cout<<"===================Intel MKL code====================="<<endl;
	cout<<"Elapsed Time: "<<time_ms<<" ms"<<endl;
	cout<<2.*1e-6*M*N*K/time_ms<<endl<<endl; 



	if(compare_matrix(C, C_mkl, M, N))
	{
		cout<<"Right kernel code!"<<endl; 
	}
	else
	{
		cout<<"Wrong kernel code!"<<endl; 
	}


	free(A); 
	free(B); 
	free(C); 
	free(C_mkl);




	
	return 0; 
}