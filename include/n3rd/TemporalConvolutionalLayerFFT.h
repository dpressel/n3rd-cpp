//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERFFT_H__
#define __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERFFT_H__

#include <sgdtk/Tensor.h>
#include "n3rd/Layer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"
#include "n3rd/FFTOps.h"

namespace n3rd
{

    class TemporalConvolutionalLayerFFT : public Layer
    {
    public:

        FFTOps conv;
        // Input is Number of frames x frame width (num feature maps)
        sgdtk::Tensor input;

        TemporalConvolutionalLayerFFT()
        {

        }

        TemporalConvolutionalLayerFFT(int nK, int kL, int kW, int embeddingSize);

        sgdtk::Tensor& forward(const sgdtk::Tensor& z);


        sgdtk::Tensor& backward(sgdtk::Tensor& chainGrad, double y);

        std::string getType() const
        { return "TemporalConvolutionalLayerFFT"; }
    };


}
#endif