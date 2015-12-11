//
// Created by Daniel on 9/25/2015.
//

#ifndef __N3RD_CPP_NEURALNETMODELFACTORY_H__
#define __N3RD_CPP_NEURALNETMODELFACTORY_H__

#include <sgdtk/ModelFactory.h>
#include <vector>
#include "n3rd/Layer.h"
#include "n3rd/NeuralNetModel.h"
namespace n3rd
{
    template<typename ModelClass = NeuralNetModel> class NeuralNetModelFactory : public sgdtk::ModelFactory
    {
        typedef sgdtk::Tensor TensorT;
        std::vector<Layer*> layerArray;
    public:
        NeuralNetModelFactory()
        {

        }
        NeuralNetModelFactory(std::vector<Layer*> layers) : layerArray(layers)
        {
        }

        /**
         * Due to inheriting the interface from SGDTk, we have an unused wlength variable, which is safely ignored.
         * @param params Settings, ignored for now but will likely be used heavily in the future
         * @return A NeuralNetModel
         */
        virtual sgdtk::Model* newInstance(void* p) const throw(sgdtk::Exception)
        {
            bool scale = false;


            if (layerArray[layerArray.size() - 1]->getType() == "SigmoidLayer")
            {
                scale = true;
            }
            return new ModelClass(layerArray, scale);
        }

        NeuralNetModelFactory& addLayer(Layer* layer)
        {
            layerArray.push_back(layer);
            return *this;
        }
    };

}

#endif
