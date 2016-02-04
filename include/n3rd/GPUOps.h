//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_GPUOPS_H__
#define __N3RD_CPP_GPUOPS_H__

#include <cuda_runtime.h>


void n3rdgTanhForward(double *dX, double *dOutput, int n);
void n3rdgTanhBackward(double *dChainGrad, double *dOutput, double *dGrads, int n);
void n3rdgMaxOverTimeForward(double* dX, double* dOutput, int* dIdx, int M, int N);
void n3rdgMaxOverTimeBackward(double* dChainGrad, int *dOrigin, double* dGrads, int M);
void n3rdgAdagradWeightUpdates(double *weights, double *weightGrads, double *gg, float eta, float lambda, int N);
void n3rdgBiasUpdates(double* biasParams, double* biasGrads, float eta, int N);
void n3rdgTranspose(double *outputMx, double* inputMx, int height, int width);
void n3rdgWrapGrad(double* unwrapped, double* grads, int kL, int kW, int oT);
void n3rdgWrapGrad2(double *unwrapped, double* grads, const int kL, int kH, int kW, int iH, int iW);
void n3rdgUnwrapInput(double *x, double* unwrapped, int kL, int kW, int iT);
void n3rdgUnwrapInput2(double* x, double* unwrapped, int kL, int kH, int kW, int iH, int iW);
void n3rdgMaxPoolingForward(double* outputMx, int* originMx, double* inputMx, int height, int width, int dh, int dw);

#endif
