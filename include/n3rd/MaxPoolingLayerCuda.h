#ifndef __N3RD_CPP_MAXPOOLINGLAYERCUDA_H__
#define __N3RD_CPP_MAXPOOLINGLAYERCUDA_H__

#include <sgdtk/Types.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include "n3rd/AbstractLayer.h"
#include "sgdtk/CudaTensor.h"
#include "n3rd/GPUOps.h"
#include <algorithm>

namespace n3rd
{


    class MaxPoolingLayerCuda : public AbstractLayer<sgdtk::CudaTensor>
    {

        sgdtk::CudaArray<int> origin;
        std::vector<int> inputDims;
        int dh;
        int dw;
    public:




        /**
         * Default Ctor, used prior to rehydrating model from file
         */
        MaxPoolingLayerCuda()
        {

        }


        explicit MaxPoolingLayerCuda(int downSampleHeight, int downSampleWidth, std::vector<int> inDims) :
                dh(downSampleHeight), dw(downSampleWidth), inputDims(inDims)
        {
            grads.resize(inputDims);
            output.resize({inputDims[0],
                           (int)std::ceil(inputDims[1]/(double)dh),
                           (int)std::ceil(inputDims[2]/(double)dw)});
            origin.resize(output.size());

        }

        const sgdtk::CudaArray<int>& getOrigin() const { return origin; }

        sgdtk::TensorI& forward(const sgdtk::TensorI& x);

        // Since the output and input are the same for the max value, we can just apply the
        // max-pool value from the output

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "MaxPoolingLayerCuda"; }
    };

}

#endif
