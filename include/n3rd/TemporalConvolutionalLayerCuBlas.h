#ifndef __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERCUBLAS_H__
#define __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERCUBLAS_H__

#include <sgdtk/Tensor.h>
#include "n3rd/Layer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"
#include <sgdtk/CudaTensor.h>

namespace n3rd
{

    class TemporalConvolutionalLayerCuBlas : public Layer
    {
        void reorderOutput(sgdtk::Tensor& unwrapped);
        void unwrapGradFromNextLayer(const sgdtk::Tensor& chainGrad, sgdtk::Tensor& unwrapped);
        void unwrapInput(const sgdtk::Tensor& x);
        void wrapGrad(const sgdtk::Tensor& unwrapped);
        int nK;
        int kL;
        int kW;
        int embedSz;
        int numFrames;
        //sgdtk::Tensor input;
        sgdtk::Tensor unwrappedInput;

        sgdtk::CudaTensor dWeights;
        sgdtk::CudaTensor dWeightGrads;
        sgdtk::CudaTensor dUnwrappedInput;
        //sgdtk::CudaTensor dGrads;

    public:


        // Input is Number of frames x frame width (num feature maps)
        sgdtk::Tensor z;

        // Output is Number of frames x num feature maps


        ///Tensor gradsW;
        ///Tensor grads;
        //int Current = 0;

        TemporalConvolutionalLayerCuBlas()
        {

        }

        TemporalConvolutionalLayerCuBlas(int nK, int kL, int kW);
        ~TemporalConvolutionalLayerCuBlas()
        {

        }


        sgdtk::Tensor& forward(const sgdtk::Tensor& input);


        sgdtk::Tensor& backward(sgdtk::Tensor& chainGrad, double y);

        std::string getType() const
        { return "TemporalConvolutionalLayerCuBlas"; }
    };


}
#endif