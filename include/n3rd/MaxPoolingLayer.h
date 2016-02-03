#ifndef __N3RD_CPP_MAXPOOLINGLAYER_H__
#define __N3RD_CPP_MAXPOOLINGLAYER_H__

#include <sgdtk/Types.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include "n3rd/AbstractLayer.h"
#include "sgdtk/Tensor.h"
#include <algorithm>

namespace n3rd
{



/**
 * Max over time pooling will take the max pixel from each feature map, and spit out a result.
 * So, if you have featureMapSize 300 for example, and you do  max over time pooling on any N-length
 * signal, you are going to get only 300 outputs.  This type of pooling is very fast, but doesnt preserve
 *
 *
 * @author dpressel
 */
    class MaxPoolingLayer : public AbstractLayer<>
    {

        std::vector<int> origin;
        std::vector<int> inputDims;
        int dh;
        int dw;
    public:




        /**
         * Default Ctor, used prior to rehydrating model from file
         */
        MaxPoolingLayer()
        {

        }


        explicit MaxPoolingLayer(int downSampleHeight, int downSampleWidth, std::vector<int> inDims) :
                dh(downSampleHeight), dw(downSampleWidth), inputDims(inDims)
        {
            grads.resize(inputDims);
            output.resize({inputDims[0],
                           (int)std::ceil(inputDims[1]/(double)dh),
                           (int)std::ceil(inputDims[2]/(double)dw)});
            origin.resize(output.size());

        }

        const std::vector<int>& getOrigin() const { return origin; }

        sgdtk::TensorI& forward(const sgdtk::TensorI& x);

        // Since the output and input are the same for the max value, we can just apply the
        // max-pool value from the output

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "MaxPoolingLayer"; }
    };

}

#endif
