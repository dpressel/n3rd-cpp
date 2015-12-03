#ifndef __N3RD_CPP_FULLYCONNECTEDLAYERCUBLAS_H__
#define __N3RD_CPP_FULLYCONNECTEDLAYERCUBLAS_H__

#include "n3rd/Layer.h"
#include <sgdtk/Tensor.h>
#include <cmath>
#include <cstdlib>
#include <sgdtk/DenseVectorN.h>
#include <sgdtk/CudaTensor.h>

namespace n3rd
{
    class FullyConnectedLayerCuBlas : public Layer
    {


        int outputLength;
        int inputLength;

        sgdtk::CudaTensor dWeights;
        sgdtk::CudaTensor dWeightGrads;
        sgdtk::CudaTensor dOutput;
        sgdtk::CudaTensor dGrads;
        sgdtk::CudaTensor dInput;

    public:
        /**
         * Empty constructor (for reincarnating models)
         */
        FullyConnectedLayerCuBlas()
        {

        }

        /**
         * Constructor, with given outputLength and input length
         * @param outputLength Output length
         * @param inputLength Input length
         */

        FullyConnectedLayerCuBlas(int outputLength, int inputLength);

        /**
         * Forward prop
         * @param x
         * @return
         */
        sgdtk::Tensor& forward(const sgdtk::Tensor& input);

        /**
         * Do backprop
         * @param outputLayerGrad layers above's deltas
         * @param y Label
         * @return The deltas for this layer
         */
        sgdtk::Tensor& backward(sgdtk::Tensor& outputLayerGrad, double y);

        int getOutputLength() const
        {
            return outputLength;
        }


        void setOutputLength(int outputLength)
        {
            this->outputLength = outputLength;
        }


        int getInputLength() const
        {
            return inputLength;
        }


        void setInputLength(int inputLength)
        {
            this->inputLength = inputLength;
        }
        std::string getType() const { return "FullyConnectedLayerCuBlas"; }

    };
}

#endif