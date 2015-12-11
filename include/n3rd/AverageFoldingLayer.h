//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_AVERAGEFOLDINGLAYER_H__
#define __N3RD_CPP_AVERAGEFOLDINGLAYER_H__

#include "n3rd/AbstractLayer.h"
#include <sgdtk/DenseVectorN.h>
#include <sgdtk/VectorN.h>
#include <sgdtk/Tensor.h>
namespace n3rd
{
    class AverageFoldingLayer : public AbstractLayer<>
    {

        int embeddingSz;
        int featureMapSz;
        int numFrames;
        int k;
    public:
        AverageFoldingLayer()
        {

        }

        AverageFoldingLayer(int numFeatureMaps, int embedSz, int kFolds) : featureMapSz(numFeatureMaps),
        embeddingSz(embedSz), k(kFolds)
        {

        }


        sgdtk::TensorI& forward(const sgdtk::TensorI& x);

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);


        int getEmbeddingSz() const
        {
            return embeddingSz;
        }

        void setEmbeddingSz(int embeddingSz)
        {
            this->embeddingSz = embeddingSz;
        }

        int getFeatureMapSz() const
        {
            return featureMapSz;
        }

        void setFeatureMapSz(int featureMapSz)
        {
            this->featureMapSz = featureMapSz;
        }

        std::string getType() const { return "AverageFoldingLayer"; }
    };
}
#endif
