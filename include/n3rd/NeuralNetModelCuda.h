#ifndef __N3RD_CPP_NEURALNETMODELCUDA_H__
#define __N3RD_CPP_NEURALNETMODELCUDA_H__

#include "n3rd/Layer.h"
#include <sgdtk/Model.h>
#include "n3rd/NeuralNetModel.h"
#include <sgdtk/CudaTensor.h>
#include <sgdtk/FeatureVector.h>
#include "n3rd/GPUOps.h"
#include <sgdtk/GPU.h>
#include <cmath>
#include <iostream>

namespace n3rd
{


    /**
     * Flexible plug-n-play Layered Neural network model trained with Adagrad
     *
     * This model is basically a wrapper around a set of Layers as building blocks, similar to Torch, and trained
     * using Adagrad (for now anyway)
     *
     * For simplicity, serialization is done using Jackson to JSON, though we could use the fast serialization provided
     * in SGDTk in the future if its helpful.
     *
     * @author dpressel
     */
    class NeuralNetModelCuda : public NeuralNetModel
    {
    protected:
        sgdtk::TensorI* makeLossTensor(double dLoss)
        {
            sgdtk::Tensor cpuTensor({dLoss}, {1});

            return new sgdtk::CudaTensor(cpuTensor);

        }


        virtual void updateBiasWeights(Layer *layer, double eta) const;

        virtual void adagradUpdate(Layer *layer, double eta, double lambda) const;

    public:
        /**
         * Default constructor, needed to reincarnate models
         */
        NeuralNetModelCuda()
        {
        }

        /**
         * Constructor supporting a stack of layers, and an argument of whether to center the output at zero and scale
         * it between -1 and 1.  This detail is encapsulated with the NeuralNetModelFactory is the preferred way to create a
         * NeuralNetModel
         *
         * @param layers A stack of network layers
         * @param scaleOutput Scale and center data?
         */
        NeuralNetModelCuda(std::vector<Layer*> layers, bool scaleOutput) : NeuralNetModel(layers, scaleOutput)
        {


        }
        virtual ~NeuralNetModelCuda()
        {
        }

        /**
         * Give back the the output layer of the network, scaling if required
         *
         * @param fv Feature vector
         * @return Return an array of scores.  In the simple binary case, we get a single value back.  In the case of a
         * log softmax type output, we get an array of values, where the index into the array is (label - 1).  The highest
         * score is then the prediction
         */

        std::vector<double> score(const sgdtk::FeatureVector* fv);

        // TODO
        void load(std::string file)
        {

        }

// TODO
        void save(std::string file)
        {

        }


    };

}

#endif
