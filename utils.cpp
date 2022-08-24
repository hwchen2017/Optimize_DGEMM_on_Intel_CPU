#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "utils.h"
using namespace std; 

void random_initial_matrix(double* A, int m, int n)
{

	srand(time(NULL)); 

	for(int i=0;i<m*n;i++)
		A[i] = (rand()/(double)RAND_MAX -0.5) * 100.0; 
}


bool compare_matrix(double *C, double *C_mkl, int m, int n)
{

	for(int i=0;i<m*n;i++)
		if(fabs(C_mkl[i] - C[i]) >= 1e-6)
		{
			cout<<i<<"th element: "<<C_mkl[i]<<"  "<<C[i]<<endl; 
			return false; 
		}

	return true; 
}

