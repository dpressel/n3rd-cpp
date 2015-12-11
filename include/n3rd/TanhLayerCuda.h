//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_TANHLAYERCUDA_H__
#define __N3RD_CPP_TANHLAYERCUDA_H__

#include "n3rd/AbstractLayer.h"
#include <sgdtk/VectorN.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include <sgdtk/TensorI.h>
#include <sgdtk/Tensor.h>
#include <sgdtk/CudaTensor.h>
#include "n3rd/GPUOps.h"

namespace n3rd
{
    class TanhLayerCuda : public AbstractLayer<sgdtk::CudaTensor>
    {
        ///sgdtk::CudaTensor dOutput;
        ///sgdtk::CudaTensor dGrads;
    public:

        TanhLayerCuda() {}
        ~TanhLayerCuda() {}

        sgdtk::TensorI& forward(const sgdtk::TensorI& input)
        {
            const sgdtk::CudaTensor& dInput = (const sgdtk::CudaTensor&)input;
            int n = dInput.size();
            ///sgdtk::CudaTensor dInput(inputT);
            output.resize(dInput.dims);
            ////grads.resize(dInput.dims);
            grads.resize(dInput.dims);
            n3rdgTanhForward(dInput.d, output.d, n);
            ///dOutput.toCPU(output);
            return output;
        }

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y)
        {
            const sgdtk::CudaTensor& dChainGrad = (const sgdtk::CudaTensor&)chainGrad;
            int n = chainGrad.size();
            ///sgdtk::Tensor chainGradT(dChainGrad.dims);

            n3rdgTanhBackward(dChainGrad.d, output.d, grads.d, n);

            ////dGrads.toCPU(grads, false);
            return grads;
        }

        std::string getType() const { return "TanhLayerCuda"; }
    };

}
#endif
