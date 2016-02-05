//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_MAXOVERTIMEPOOLINGLAYER_H__
#define __N3RD_CPP_MAXOVERTIMEPOOLINGLAYER_H__

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
    class MaxOverTimePoolingLayer : public AbstractLayer<>
    {

        int featureMapSz;
        int numFrames;
        std::vector<int> origin;
    public:


        enum {DS_MIN = -1000000 };

        /**
         * Default Ctor, used prior to rehydrating model from file
         */
        MaxOverTimePoolingLayer()
        {

        }


        explicit MaxOverTimePoolingLayer(int numFeatureMaps) :
                featureMapSz(numFeatureMaps)
        {
            output.resize({featureMapSz});
            origin.resize(featureMapSz);
        }

        int getFeatureMapSz()
        {
            return featureMapSz;
        }

        void setFeatureMapSz(int numFeatureMaps)
        {
            featureMapSz = numFeatureMaps;
        }

        const std::vector<int>& getOrigin() const { return origin; }

        sgdtk::TensorI& forward(const sgdtk::TensorI& x);

        // Since the output and input are the same for the max value, we can just apply the
        // max-pool value from the output

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "MaxOverTimePoolingLayer"; }
    };

}

#endif
