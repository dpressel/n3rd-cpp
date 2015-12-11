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

#endif
