#ifndef __N3RD_CPP_FULLYCONNECTEDLAYERBLAS_H__
#define __N3RD_CPP_FULLYCONNECTEDLAYERBLAS_H__

#include "n3rd/AbstractLayer.h"
#include <sgdtk/Tensor.h>
#include <sgdtk/TensorI.h>
#include <cmath>
#include <cstdlib>
#include <sgdtk/DenseVectorN.h>


namespace n3rd
{
    class FullyConnectedLayerBlas : public AbstractLayer<>
    {

        sgdtk::Tensor z;
        int outputLength;
        int inputLength;

    public:
        /**
         * Empty constructor (for reincarnating models)
         */
        FullyConnectedLayerBlas()
        {

        }

        /**
         * Constructor, with given outputLength and input length
         * @param outputLength Output length
         * @param inputLength Input length
         */

        FullyConnectedLayerBlas(int outputLength, int inputLength);

        /**
         * Forward prop
         * @param x
         * @return
         */
        sgdtk::TensorI& forward(const sgdtk::TensorI& input);

        /**
         * Do backprop
         * @param outputLayerGrad layers above's deltas
         * @param y Label
         * @return The deltas for this layer
         */
        sgdtk::TensorI& backward(sgdtk::TensorI& outputLayerGrad, double y);

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
        std::string getType() const { return "FullyConnectedLayerBlas"; }

    };
}

#endif