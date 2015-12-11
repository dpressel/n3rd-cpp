//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_FFTOPS_H__
#define __N3RD_CPP_FFTOPS_H__

#include <vector>
#include <sgdtk/Tensor.h>
#include <fftw3.h>

namespace n3rd
{
    class FFTOps
    {
        fftw_complex *xy;
        std::map<int, std::pair<fftw_plan, fftw_plan>> plans;
        int currentWidth;
    public:


        static int nextPowerOf2(int n);

        FFTOps() : xy(nullptr), currentWidth(0)
        {

        }

        void destroyAllPlans();

        ~FFTOps() {
            destroyAllPlans();
            if (xy) fftw_free(xy);
        }
        void fftfilt1(const sgdtk::Tensor& data, const sgdtk::Tensor& kernels, const sgdtk::Tensor& biases, sgdtk::Tensor& output, bool corr = true);
    };
}

#endif