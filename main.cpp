#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdbool>
#include <chrono>
#include <mkl.h>
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
        case 0: dgemm_kernel_t3(M, N, K, alpha, A, M, B, K, beta, C, M); 
		break;
// 		case 9: dgemm_kernel_v9(M, N, K, alpha, A, M, B, K, beta, C, M); 
// 		break;       
	}
}


int main(int argc, char *argv[])
{

	int kernel_num = 0; 

	if(argc == 2) kernel_num=atoi(argv[1]);

	// cout<<kernel_num<<endl; 

	vector<int> ss; 

	for(int i=1;i<=25;i++)
		ss.push_back(i*128); 



	int M, N, K;

	double *A, *B, *C, *C_mkl;

	double alpha = 1.0, beta = 0.0; 
	ss[3] = 1024; 

	M = ss[3], N = ss[3], K = ss[3]; 

	printf("Matrix Size: %dX%d, %dX%d\n\n", M, K, K, N ); 

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

	auto start = std::chrono::high_resolution_clock::now();
	test_dgemm_kernel(kernel_num, M, N, K, alpha, A, B, beta, C); 

	auto end = std::chrono::high_resolution_clock::now(); 
	auto elapsed = end - start;

	double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() * 1e-3;

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


	start = std::chrono::high_resolution_clock::now();

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C_mkl, M);

	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start; 

	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() * 1e-3 ;
// 	cout<<"Time: "<<time_ms<<" ms"<<endl;


	start = std::chrono::high_resolution_clock::now();

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C_mkl, M);

	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start; 

	time_ms = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() * 1e-3 ;

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