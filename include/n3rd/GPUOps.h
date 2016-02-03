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
void n3rdgWrapGrad(double* unwrapped, double* grads, int L, int kW, int oT);
void n3rdgUnwrapInput(double *x, double* unwrapped, int L, int kW, int iT);
void n3rdgMaxPoolingForward(double* outputMx, int* originMx, double* inputMx, int height, int width, int dh, int dw);

#endif
