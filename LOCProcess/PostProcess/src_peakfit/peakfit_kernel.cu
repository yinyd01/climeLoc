#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "DLL_Macros.h"
#include "CUDA_Utils.cuh"
#include "definitions.h"
#include "Cholesky.cuh"
#include "functionals.cuh"

#define BLCKSZ 64

/*! @brief Fit a 2D matrix with a bivariate Gaussian distribution
    @param[in]      NFits:      int, number of fits
    @param[in]      imHeight:   int, size of the matrix (y-axis)
    @param[in]      imWidth:    int, size of the matrix (x-axis)
    @param[in]      g_data:     (NFits * imHeight * imWidth) float, the 2D data
    @param[in]      tinv095:    float, matlab result of tinv(0.975, df) or scipy result of scipy.stats.t.ppf(0.975, df), df = imHeight * imWidth - VNUM
    @param[in]      MAX_ITERS:  int, number of maximum iterations
    @param[out]     g_xvec:     (NFits * VNUM) float, [[x, y, sx, sy, rho, Intensity, bkg],...] the parameters of the bivariate Gaussian model
    @param[out]     g_Loss:     (NFits) float, the chi-2 of each fit
    @param[out]     g_crlb:     (NFits * VNUM) float array, the crlb of each parameter of the bivariate Gaussian model
    @param[out]     g_ci:       (NFits * VNUM) float array, the 95% confidence interval for g_xvec

    @param[local]   pvec:       (VNUM) float array, the iterative change of the xvec
    @param[local]   maxJump:    (VNUM) float array, the maxJump of pvec at each iteration
    @param[local]   grad:       (VNUM) float array, the gradience of the Least Square over each parameter of the xvec, respectively
    @param[local]   Hessian:    (VNUM * VNUM) float array, the Hessian of the Least Square over each parameter of the xvec, pairwisely
    @param[local]   JJ:         (VNUM * VNUM) float array, the Jacobian * Jacobian 
*/
// <<<ceil(NFits/BLCKSZ), BLCKSZ = 64>>>
__global__ void PeakFit_Gauss(int NFits, int imHeight, int imWidth, float* g_data, float tinv095, int MAX_ITERS, 
    float* g_xvec, float* g_Loss, float* g_crlb, float* g_ci)
{
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    float *imdata = nullptr; 
    
    float data = 0.0f, model = 0.0f, Loss = 0.0f, Loss_new = 0.0f, dLoss = 0.0f, deriv1[VNUM] = { 0.0f };
    float lambda = INIT_LAMBDA, mu = 1.0f;

    float xvec[VNUM] = { 0.0f }, pvec[VNUM] = { 0.0f }, maxJump[VNUM] = { 0.0f };
    float xvec_new[VNUM] = { 0.0f }, pvec_new[VNUM] = { 0.0f }, maxJump_new[VNUM] = { 0.0f };
    float grad[VNUM] = { 0.0f }, Hessian[VNUM*VNUM] = { 0.0f }, JJ[VNUM*VNUM] = { 0.0f }, L[VNUM*VNUM] = { 0.0f }; 

    int decomp_flag = 0, pxID = 0, iter = 0, i = 0, j = 0;
    float df = (float)(imHeight * imWidth - VNUM);

    if (Idx < NFits) {
        imdata = g_data + Idx * imHeight * imWidth;

        // Initialization
        _INIT_PROFILE(imHeight, imWidth, imdata, xvec);

        maxJump[0] = max(1.0f, 0.1f*(float)imWidth);
        maxJump[1] = max(1.0f, 0.1f*(float)imHeight);
        maxJump[2] = max(0.3f, 0.05f*(float)imWidth);
        maxJump[3] = max(0.3f, 0.05f*(float)imHeight);
        maxJump[4] = max(0.05f, xvec[4]);
        maxJump[5] = max(100.0f, xvec[5]);
        maxJump[6] = max(20.0f, xvec[6]);

        for (i = 0; i < VNUM; i++)
            pvec[i] = INIT_ERR;
        
        memset(grad, 0, VNUM * sizeof(float));
        memset(Hessian, 0, VNUM * VNUM * sizeof(float));
        for (pxID = 0, Loss = 0.0f; pxID < imHeight*imWidth; pxID++) {
            data = imdata[pxID];
            _fconstruct(pxID, imHeight, imWidth, xvec, &model, deriv1);
            Loss += (model - data) * (model - data);

            for (i = 0; i < VNUM; i++)
                grad[i] += 2.0f * (model - data) * deriv1[i];
            for (i = 0; i < VNUM; i++)
                for (j = i; j < VNUM; j++)
                    Hessian[i * VNUM + j] += 2.0f * deriv1[i] * deriv1[j];
        }
        for (i = 0; i < VNUM; i++)
            for (j = i; j < VNUM; j++)
                Hessian[j * VNUM + i] = Hessian[i * VNUM + j];
        
        // L-M iteration
        for (iter = 0; iter < MAX_ITERS; iter++) {
            decomp_flag = Cholesky_decomp(Hessian, VNUM, L);
            if (decomp_flag == 0) {
                mu = max((1.0f + lambda * SCALE_UP) / (1.0f + lambda), 1.3f);
                for (i = 0; i < VNUM; i++)
                    Hessian[i * VNUM + i] *= mu;
                lambda *= SCALE_UP;
                continue;
            }
            
            Cholesky_solve(L, grad, VNUM, pvec_new);
            memcpy(maxJump_new, maxJump, VNUM * sizeof(float));
            for (i = 0; i < VNUM; i++) {
                if (pvec_new[i] / pvec[i] < -0.5f)
                    maxJump_new[i] *= 0.5f;
                pvec_new[i] /= 1.0f + fabs(pvec_new[i] / maxJump_new[i]);
                xvec_new[i] = xvec[i] - pvec_new[i];
            }
            xvec_new[0] = min(max(xvec_new[0], 0.1f*(float)imWidth), 0.9f*(float)imWidth);
            xvec_new[1] = min(max(xvec_new[1], 0.1f*(float)imHeight), 0.9f*(float)imHeight);
            xvec_new[2] = min(max(xvec_new[2], 0.5f), 10.0f);
            xvec_new[3] = min(max(xvec_new[3], 0.5f), 10.0f);
            xvec_new[4] = min(max(xvec_new[4], 0.0f), 0.99f);
            xvec_new[5] = min(max(xvec_new[5], IMIN), IMAX);
            xvec_new[6] = min(max(xvec_new[6], BGMIN), BGMAX);  

            for (pxID = 0, Loss_new = 0.0f; pxID < imHeight*imWidth; pxID++) {
                data = imdata[pxID];
                _fconstruct(pxID, imHeight, imWidth, xvec_new, &model, NULL);
                Loss_new += (model - data) * (model - data);
            }

            if (Loss_new > ACCEPTANCE * Loss) {
                mu = max((1.0f + lambda * SCALE_UP) / (1.0f + lambda), 1.3f);
                for (i = 0; i < VNUM; i++)
                    Hessian[i * VNUM + i] *= mu;
                lambda *= SCALE_UP;
                continue;
            }

            if (Loss_new < Loss) {
                mu = 1.0f + SCALE_DOWN * lambda;
                lambda *= SCALE_DOWN;
            }

            memcpy(xvec, xvec_new, VNUM * sizeof(float));
            memcpy(maxJump, maxJump_new, VNUM * sizeof(float));
            memcpy(pvec, pvec_new, VNUM * sizeof(float));
            memset(grad, 0, VNUM * sizeof(float));
            memset(Hessian, 0, VNUM * VNUM * sizeof(float));
            for (pxID = 0; pxID < imHeight*imWidth; pxID++) {
                data = imdata[pxID];
                _fconstruct(pxID, imHeight, imWidth, xvec, &model, deriv1);

                for (i = 0; i < VNUM; i++)
                    grad[i] += 2.0f * (model - data) * deriv1[i];
                for (i = 0; i < VNUM; i++)
                    for (j = i; j < VNUM; j++)
                        Hessian[i * VNUM + j] += 2.0f * deriv1[i] * deriv1[j];
                for (i = 0; i < VNUM; i++)
                    for (j = i; j < VNUM; j++)
                        JJ[i * VNUM + j] += 4.0f * (model - data) * deriv1[i] * (model - data) * deriv1[j];
            }
            for (i = 0; i < VNUM; i++)
                for (j = i; j < VNUM; j++)
                    Hessian[j * VNUM + i] = Hessian[i * VNUM + j];
            for (i = 0; i < VNUM; i++)
                for (j = i; j < VNUM; j++)
                    JJ[j * VNUM + i] = JJ[i * VNUM + j];
            
            dLoss = Loss - Loss_new;
            Loss = Loss_new;
            if (dLoss <= Loss * OPTTOL)
                break;
        }
        
        // Finalization (var * inv(Hessian) is the Fisher Information Matrix for Least Square)
        memcpy(g_xvec + Idx * VNUM, xvec, VNUM * sizeof(float));
        g_Loss[Idx] = Loss;
        
        decomp_flag = Cholesky_decomp(Hessian, VNUM, L);
        if (decomp_flag == 1) {
            Cholesky_invert(L, VNUM, Hessian); // use Hessian to store its inverse
            for (i = 0; i < VNUM; i++)
                g_crlb[Idx * VNUM + i] = Loss / (float)(imHeight * imWidth) * Hessian[i * VNUM + i];
        }
        else
            for (i = 0; i < VNUM; i++)
                g_crlb[Idx * VNUM + i] = -1.0f;
        
        decomp_flag = Cholesky_decomp(JJ, VNUM, L);
        if (decomp_flag == 1) {
            Cholesky_invert(L, VNUM, JJ); // use JJ to store its inverse
            for (i = 0; i < VNUM; i++)
                if (JJ[i * VNUM + i] > 0.0f)
                    g_ci[Idx * VNUM + i] = 2.0f * tinv095 * sqrt(Loss / df * JJ[i * VNUM + i]);
                else
                    g_ci[Idx * VNUM + i] = -1.0f; 
        }
        else
            for (i = 0; i < VNUM; i++)
                g_ci[Idx * VNUM + i] = -1.0f;
    }
    __syncthreads();
    return;
}



static inline int intceil(int n, int m) { return (n - 1) / m + 1; }
static inline dim3 dimGrid1(int n, int blocksz) { dim3 gridsz(intceil(n, blocksz), 1, 1); return gridsz; }
static inline dim3 dimBlock1(int n) { dim3 blocksz(n, 1, 1); return blocksz; }


CDLL_EXPORT void kernel(int NFits, int imHeight, int imWidth, float* h_data, float tinv095, int MAX_ITERS, 
    float* h_xvec, float* h_Loss, float* h_crlb, float* h_ci)
{
    float *d_data = nullptr, *d_xvec = nullptr, *d_Loss = nullptr, *d_crlb = nullptr, *d_ci = nullptr;
    
    // allocate space on device
    cudasafe(cudaMalloc(&d_data, NFits * imHeight * imWidth * sizeof(float)), "malloc d_data", __LINE__);
    cudasafe(cudaMalloc(&d_xvec, NFits * VNUM * sizeof(float)), "malloc d_xvec", __LINE__);
    cudasafe(cudaMalloc(&d_Loss, NFits * sizeof(float)), "malloc d_Loss", __LINE__);
    cudasafe(cudaMalloc(&d_crlb, NFits * VNUM * sizeof(float)), "malloc d_crlb", __LINE__);
    cudasafe(cudaMalloc(&d_ci, NFits * VNUM * sizeof(float)), "malloc d_ci", __LINE__);

    // transfer data from host to device
    cudasafe(cudaMemcpy(d_data, h_data, NFits * imHeight * imWidth * sizeof(int), cudaMemcpyHostToDevice), "transfer h_data to device", __LINE__);
    
    // launch kernel
    PeakFit_Gauss<<<dimGrid1(NFits, BLCKSZ), dimBlock1(BLCKSZ)>>>(NFits, imHeight, imWidth, d_data, tinv095, MAX_ITERS, d_xvec, d_Loss, d_crlb, d_ci);
    
    // transfer results from device to host
    cudasafe(cudaMemcpy(h_xvec, d_xvec, NFits * VNUM * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_xvec to host", __LINE__);
    cudasafe(cudaMemcpy(h_Loss, d_Loss, NFits * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_Loss to host", __LINE__);
    cudasafe(cudaMemcpy(h_crlb, d_crlb, NFits * VNUM * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_crlb to host", __LINE__);
    cudasafe(cudaMemcpy(h_ci, d_ci, NFits * VNUM * sizeof(float), cudaMemcpyDeviceToHost), "transfer d_ci to host", __LINE__);

    // free spaces
    cudasafe(cudaFree(d_data), "free d_data on device", __LINE__);
    cudasafe(cudaFree(d_xvec), "free d_xvec on device", __LINE__);
    cudasafe(cudaFree(d_Loss), "free d_Loss on device", __LINE__);
    cudasafe(cudaFree(d_crlb), "free d_crlb on device", __LINE__);
    cudasafe(cudaFree(d_ci), "free d_ci on device", __LINE__);
    return;
}
