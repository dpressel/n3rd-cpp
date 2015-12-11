//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_FILTEROPS_H__
#define __N3RD_CPP_FILTEROPS_H__

#include <vector>
#include <sgdtk/Tensor.h>
#include <fftw3.h>

namespace n3rd
{
    class FilterOps
    {
    public:
        static void corr2Weights(const sgdtk::Tensor& x, const sgdtk::Tensor& ygrad, sgdtk::Tensor& weightGrads);

        static void corr1Weights(const sgdtk::Tensor& x, const sgdtk::Tensor& ygrad, sgdtk::Tensor& weightGrads);

        static void corr1(const sgdtk::Tensor& data, const sgdtk::Tensor& kernels, const sgdtk::Tensor &biases, sgdtk::Tensor& output);

        static void conv1(const sgdtk::Tensor& data, const sgdtk::Tensor& kernels, const sgdtk::Tensor &biases, sgdtk::Tensor& output);

        static void conv2(const sgdtk::Tensor& data, const sgdtk::Tensor& kernels, const sgdtk::Tensor &biases, sgdtk::Tensor& output);

        static void corr2(const sgdtk::Tensor& data, const sgdtk::Tensor& kernels, const sgdtk::Tensor &biases, sgdtk::Tensor& output);



        static void fftfilt1(const sgdtk::Tensor& data, const sgdtk::Tensor& kernels, const sgdtk::Tensor &biases, sgdtk::Tensor& output, bool corr = true);
    };
}

#endif