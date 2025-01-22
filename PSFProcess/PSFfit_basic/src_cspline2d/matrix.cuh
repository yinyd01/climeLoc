#pragma once
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "definitions.h"

#ifndef MATRIX_CUH
#define MATRIX_CUH


/*!
	@brief	Choleskey decomposition of a matrix
	@param[in]	Mat:	[n * n], the inpur matrix for decomposition
	@param[in]	n: 		number of rows(cols)
	@param[out]	L:		[n * n], the decomposed Lower-triangular matrix
	@return (choleskey decomposible) ? 1 : 0 
*/
__device__ static char Cholesky_decomp(float* Mat, int n, float* L)
{
	char info = 1;
	int i = 0, j = 0, k = 0;
	float dum = 0.0f;

	for (i = 0; i < n; i++)
		for (j = 0; j <= i; j++) {
			for (k = 0, dum = 0.0f; k < j; k++)
				dum += L[i * n + k] * L[j * n + k];

			if (i == j)
				if (Mat[i * n + i] - dum <= 0) {
					info = 0;
					return info;
				}
				else	
					L[i * n + j] = sqrt(Mat[i * n + i] - dum);
			else
				L[i * n + j] = (1.0 / L[j * n + j] * (Mat[i * n + j] - dum));
		}
	return info;
}



/*!
	@brief	Solve Ax = b where A = LLT is symmetric and and decomposed to L via cholesky decomposition
	@param[in]	L:		[n * n] the decomposed Lower-triangular matrix
	@param[in]	b:		[n] the target vector
	@param[in]	n: 		number of rows(cols)
	@param[out] x:		[n] the solution vector 
*/
__device__ static void Cholesky_solve(float* L, float* b, int n, float* x)
{
	int i = 0, k = 0;
	
	// Forward substitution
	for (i = 0; i < n; i++) {
		x[i] = b[i];
		for (k = 0; k < i; k++)
			x[i] -= L[i * n + k] * x[k];
		x[i] /= L[i * n + i];
	}
	// Backward substitution
	for (i = n - 1; i >= 0; i--) {
		for (k = i + 1; k < n; k++)
			x[i] -= L[k * n + i] * x[k];
		x[i] /= L[i * n + i];
	}
}



/*!
	@brief	Invert Mat where Mat = LLT is symmetric and and decomposed to L via cholesky decomposition
	@param[in]	L:		[n * n] the decomposed Lower-triangular matrix
	@param[in]	n: 		number of rows(cols)
	@param[out] invMat:	[n * n] the inverse of the Mat 
*/
__device__ static void Cholesky_invert(float* L, int n, float* invMat)
{
	int i = 0, j = 0, k = 0;
	
	for (j = 0; j < n; j++) {
		// Forward substitution
		for (i = 0; i < n; i++) {
			invMat[i * n + j] = (i == j) ? 1.0f : 0.0f;
			for (k = 0; k < i; k++)
				invMat[i * n + j] -= L[i * n + k] * invMat[k * n + j];
			invMat[i * n + j] /= L[i * n + i];
		}
		// Backward substitution
		for (i = n - 1; i >= 0; i--) {
			for (k = i + 1; k < n; k++)
				invMat[i * n + j] -= L[k * n + i] * invMat[k * n + j];
			invMat[i * n + j] /= L[i * n + i];
		}
	}	
}


#endif