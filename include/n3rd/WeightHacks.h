//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_WEIGHTHACKS_H__
#define __N3RD_CPP_WEIGHTHACKS_H__

#include <vector>
#include <sgdtk/FeatureVector.h>
#include "n3rd/NeuralNetModel.h"

namespace n3rd
{
    class WeightHacks
    {
    public:
        static void shuffle(std::vector<sgdtk::FeatureVector*>& instances);

        static void hack(NeuralNetModel* nnModel);
    };
}

#endif