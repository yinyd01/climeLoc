#pragma once
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "definitions.h"

#ifndef MATRIX_CUH
#define MATRIX_CUH


/*! @brief	Choleskey decomposition of a matrix
	@param[in]	Mat:	[n * n], the inpur matrix for decomposition
	@param[in]	n: 		number of rows(cols)
	@param[out]	L:		[n * n], the decomposed Lower-triangular matrix
	@return (choleskey decomposible) ? 1 : 0 
*/
__device__ static char Cholesky_decomp(float* Mat, int n, float* L)
{
	float dum = 0.0f;
	char info = 1;
	int i = 0, j = 0, k = 0;
	memset(L, 0, n * n * sizeof(float));
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



/*!	@brief	Solve Ax = b where A = LLT is symmetric and and decomposed to L via cholesky decomposition
	@param[in]	L:		[n * n] the decomposed Lower-triangular matrix
	@param[in]	b:		[n] the target vector
	@param[in]	n: 		number of rows(cols)
	@param[out] x:		[n] the solution vector 
*/
__device__ static void Cholesky_solve(float* L, float* b, int n, float* x)
{
	int i = 0, k = 0;
	memset(x, 0, n * sizeof(float));
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



/*!	@brief	Invert Mat where Mat = LLT is symmetric and and decomposed to L via cholesky decomposition
	@param[in]	L:		[n * n] the decomposed Lower-triangular matrix
	@param[in]	n: 		number of rows(cols)
	@param[out] invMat:	[n * n] the inverse of the Mat 
*/
__device__ static void Cholesky_invert(float* L, int n, float* invMat)
{
	int i = 0, j = 0, k = 0;
	memset(invMat, 0, n *n * sizeof(float));
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



/*!	@brief	check if a pixel holds the local maximum in a matrix
	@param[in]	Mat:			(n * n) float, the matrix to find local maxima
	@param[in]	n: 				int, number of rows(cols)
	@param[in]	row:			int, the row of the testing pixel
	@param[in]	col:			int, the col of the testing pixel
	@param[in]	radius:			int, the radius for local maxima check
	@return 	islocalmax		bool, true if it is a local maximum
*/
__device__ static bool _islocalmaximum(float* Mat, int n, int row, int col, int radius)
{
	const float current_value = Mat[row * n + col];
	for (int i = row - radius; i <= row + radius; i++) {
		if (i < 0 || i >= n)
			continue;
		for (int j = col - radius; j <= col + radius; j++) {
			if ( j < 0 || j >= n)
				continue;
			if (Mat[i * n + j] > current_value)
				return false;
		}
	}
	return true;
}



/*!	@brief	find the highest k local maxima
	@param[in]	Mat:			(n * n) float, the matrix to find local maxima
	@param[in]	n: 				int, number of rows(cols)
	@param[in]	k:				int, number of highest local maxima to search (k <= NMAX)
	@param[out] LocalMaxima:	(k * 3) float, [x, y, value] of each local maxima, (preallocate memory of NMAX * 3 * sizeof(float))
*/
__device__ static void find_localmax(float* Mat, int n, int k, float* LocalMaxima)
{
	const int radius = 1;
	memset(LocalMaxima, 0, NMAX * 3 * sizeof(float));
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) 
			if (_islocalmaximum(Mat, n, i, j, radius))
				for (unsigned int q = 0; q < k; q++) 
					if (Mat[i * n + j] > LocalMaxima[q * 3 + 2]) {
						for (unsigned int o = k - 1; o > q; o--) 
							memcpy(LocalMaxima + o * 3, LocalMaxima + (o - 1) * 3, 3 * sizeof(float));
						LocalMaxima[q * 3] = (float)i + 0.5f;
						LocalMaxima[q * 3 + 1] = (float)j + 0.5f;
						LocalMaxima[q * 3 + 2] = Mat[i * n + j];
                        break;
					}
}


#endif