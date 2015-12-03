#ifndef __SGDTK_GPU_H__
#define __SGDTK_GPU_H__


#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "sgdtk/Types.h"
#include "sgdtk/Exception.h"
#include "sgdtk/Tensor.h"
//#include <cudnn.h>
#include <cublas.h>

#define TOSTR_(s)   #s
#define TOSTR(s)    TOSTR_(s)
#if defined(__GNUC__)
#define COMPILER_NAME "GCC"
#define COMPILER_VER  TOSTR(__GNUC__) "." TOSTR(__GNUC_MINOR__) "." TOSTR(__GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#if _MSC_VER < 1500
#define COMPILER_NAME "MSVC_2005"
#elif _MSC_VER < 1600
#define COMPILER_NAME "MSVC_2008"
#elif _MSC_VER < 1700
#define COMPILER_NAME "MSVC_2010"
#elif _MSC_VER < 1800
#define COMPILER_NAME "MSVC_2012"
#elif _MSC_VER < 1900
#define COMPILER_NAME "MSVC_2013"
#elif _MSC_VER < 2000
#define COMPILER_NAME "MSVC_2014"
#else
#define COMPILER_NAME "MSVC"
#endif
#define COMPILER_VER  TOSTR(_MSC_FULL_VER) "." TOSTR(_MSC_BUILD)
#elif defined(__clang_major__)
#define COMPILER_NAME "CLANG"
#define COMPILER_VER  TOSTR(__clang_major__ ) "." TOSTR(__clang_minor__) "." TOSTR(__clang_patchlevel__)
#elif defined(__INTEL_COMPILER)
#define COMPILER_NAME "ICC"
#define COMPILER_VER TOSTR(__INTEL_COMPILER) "." TOSTR(__INTEL_COMPILER_BUILD_DATE)
#else
#define COMPILER_NAME "unknown"
#define COMPILER_VER  "???"
#endif

#define CUDNN_VERSION_STR  TOSTR(CUDNN_MAJOR) "." TOSTR (CUDNN_MINOR) "." TOSTR(CUDNN_PATCHLEVEL)

#define THROW_ERR(s) {                                                 \
    std::stringstream where, msg;                                      \
    where << __FILE__ << ':' << __LINE__;                              \
    msg << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;       \
    cudaDeviceReset();                                                 \
    throw sgdtk::Exception(msg.str());                                 \
}

//#define TRY_CUDNN(status) {                                            \
//    std::stringstream err;                                             \
//    if (status != CUDNN_STATUS_SUCCESS) {                              \
//      err << "CUDNN failure\nError: " << cudnnGetErrorString(status);  \
//      THROW_ERR(err.str());                                            \
//    }                                                                  \
//}

#define TRY_CUDA(status) {                                             \
    std::stringstream err;                                             \
    if (status != 0) {                                                 \
      err << "Cuda failure\nError: " << cudaGetErrorString(status);    \
      THROW_ERR(err.str());                                            \
    }                                                                  \
}

#define TRY_CUBLAS(status) {                                           \
    std::stringstream err;                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
      err << "Cublas failure\nError code " << status;                  \
      THROW_ERR(err.str());                                            \
    }                                                                  \
}

namespace sgdtk
{
    struct Globals {
        //static cudnnHandle_t gHandle;

        static cublasHandle_t gBlasHandle;
    };
    //inline void initCuDNN()
    //{
    //    TRY_CUDNN(cudnnCreate(&Globals::gHandle));
    //}
    inline void initCuBlas()
    {
        //TRY_CUBLAS(cublasInit());
        TRY_CUBLAS(cublasCreate_v2(&Globals::gBlasHandle));
    }
    //inline void doneCuDNN()
    //{
    //    TRY_CUDNN(cudnnDestroy(Globals::gHandle));
    //}
    inline void doneCuBlas()
    {
        //TRY_CUBLAS(cublasShutdown());
        TRY_CUBLAS(cublasDestroy_v2(Globals::gBlasHandle));
    }

    // These functions are so annoying to setup, that Im just simplifying them as much as possible
    template<typename T> T* createGPUArray(int nelem)
    {
        int sz = nelem * sizeof(T);
        T* gpuPtr;
        TRY_CUDA(cudaMalloc(&gpuPtr, sz));
        return gpuPtr;
    }

    template<typename T, typename HostContainer_T = std::vector<T> > void copyArrayToGPU(T* gpuPtr, HostContainer_T& hostArray)
    {
        TRY_CUDA(cudaMemcpy(gpuPtr, &hostArray[0], hostArray.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T, typename HostContainer_T = std::vector<T> > void copyArrayFromGPU(T* gpuPtr, HostContainer_T& properlySizedHostArray)
    {
        T* target = &properlySizedHostArray[0];
        int nelem = properlySizedHostArray.size();
        TRY_CUDA(cudaMemcpy(target, gpuPtr, nelem * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T, typename HostContainer_T = std::vector<T> > T* createGPUArrayFromArray(HostContainer_T& hostArray)
    {
        T* gpuPtr = createGPUArray<T>(hostArray.size());
        copyArrayToGPU(gpuPtr, hostArray);
        return gpuPtr;
    }
/*
    inline void setTensorDescriptor(cudnnTensorDescriptor_t & descr, const Tensor& hostTensor)
    {
        cudnnDataType_t dt = CUDNN_DATA_FLOAT;
        if (sizeof(Real) == 8)
        {
            dt = CUDNN_DATA_DOUBLE;
        }
        const int nDims = hostTensor.dims.size();
        int stridesArray[nDims];
        int dimsArray[nDims];



        int stride = 1;
        for (int i = nDims - 1; i >= 0; --i)
        {
            stridesArray[i] = stride;
            stride *= hostTensor.dims[i];
            dimsArray[i] = hostTensor.dims[i];
        }

        TRY_CUDNN(cudnnSetTensorNdDescriptor(descr, dt, nDims, dimsArray, stridesArray));

    }

    inline void setTensorDescriptor(cudnnTensorDescriptor_t & descr, int sizeofT, const std::vector<int>& dims)
    {
        cudnnDataType_t dt = CUDNN_DATA_FLOAT;
        if (sizeofT == 8)
        {
            dt = CUDNN_DATA_DOUBLE;
        }
        const int nDims = dims.size();
        int stridesArray[nDims];
        int dimsArray[nDims];
        int stride = 1;
        for (int i = nDims - 1; i >= 0; --i)
        {
            stridesArray[i] = stride;
            stride *= dims[i];
            dimsArray[i] = dims[i];
        }

        TRY_CUDNN(cudnnSetTensorNdDescriptor(descr, dt, nDims, dimsArray, stridesArray));

    }
    struct CuDNNFilter
    {
        cudnnFilterDescriptor_t descriptor;
        std::vector<int> dims;
        sgdtk::Real*d;

        CuDNNFilter() : d(NULL), descriptor(NULL){}
        void setup(const sgdtk::Tensor& tensor)
        {

            dims = tensor.dims;
            if (dims.size() != 4)
            {
                throw Exception("Bad dims");
            }
            int nK = dims[0];
            int kL = dims[1];
            int kH = dims[2];
            int kW = dims[3];

            std::vector<int> upsample = {1, 1};
            std::vector<int> zp = {0, 0};
            std::vector<int> filterStride = { nK * kL * kH * kW, kL * kH * kW, kH * kW, kW};
            std::vector<int> filterDims = {nK, kL, kH, kW};

            if (d)
            {
                cudaFree(d);
            }
            if (descriptor)
            {
                cudaFree(descriptor);
            }
            d = sgdtk::createGPUArrayFromArray<sgdtk::Real>(tensor);
            TRY_CUDNN(cudnnCreateFilterDescriptor(&descriptor));


            TRY_CUDNN(cudnnSetFilter4dDescriptor(descriptor, CUDNN_DATA_DOUBLE, nK, kL, kH, kW));


            //TRY_CUDNN(cudnnSetFilterNdDescriptor(descriptor, CUDNN_DATA_DOUBLE, 4, &filterDims[0]));
        }

        ~CuDNNFilter()
        {
            if (d)
            {
                cudaFree(d);
            }
            if (descriptor)
            {
                cudaFree(descriptor);
            }
        }

        void toCPU(Tensor& cpuTensor)
        {
            cpuTensor.resize(dims);
            sgdtk::copyArrayFromGPU(d, cpuTensor);
        }
    };
*/

}

#endif