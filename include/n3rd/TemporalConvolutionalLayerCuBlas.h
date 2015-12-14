#ifndef __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERCUBLAS_H__
#define __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERCUBLAS_H__

#include <sgdtk/Tensor.h>
#include "n3rd/Layer.h"
#include "n3rd/AbstractLayer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"
#include <sgdtk/CudaTensor.h>
#include "n3rd/GPUOps.h"

namespace n3rd
{

    class TemporalConvolutionalLayerCuBlas : public AbstractLayer<sgdtk::CudaTensor>
    {

        void unwrapInput(const sgdtk::CudaTensor& x);
        void wrapGrad(const sgdtk::CudaTensor& unwrapped);
        int nK;
        int kL;
        int kW;
        int embedSz;
        int numFrames;

        sgdtk::CudaTensor dUnwrappedInput;


    public:

        // Input is Number of frames x frame width (num feature maps)
        sgdtk::Tensor z;

        TemporalConvolutionalLayerCuBlas()
        {

        }

        TemporalConvolutionalLayerCuBlas(int nK, int kL, int kW);
        ~TemporalConvolutionalLayerCuBlas()
        {

        }


        sgdtk::TensorI& forward(const sgdtk::TensorI& input);


        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const
        { return "TemporalConvolutionalLayerCuBlas"; }
    };


}
#endif