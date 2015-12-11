//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERFFT_H__
#define __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERFFT_H__

#include <sgdtk/Tensor.h>
#include "n3rd/AbstractLayer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"
#include "n3rd/FFTOps.h"

namespace n3rd
{

    class TemporalConvolutionalLayerFFT : public AbstractLayer<>
    {
    public:

        FFTOps conv;
        // Input is Number of frames x frame width (num feature maps)
        sgdtk::Tensor input;

        TemporalConvolutionalLayerFFT()
        {

        }

        TemporalConvolutionalLayerFFT(int nK, int kL, int kW, int embeddingSize);

        sgdtk::TensorI& forward(const sgdtk::TensorI& z);


        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const
        { return "TemporalConvolutionalLayerFFT"; }
    };


}
#endif