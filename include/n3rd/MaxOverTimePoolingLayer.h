//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_MAXOVERTIMEPOOLINGLAYER_H__
#define __N3RD_CPP_MAXOVERTIMEPOOLINGLAYER_H__

#include <sgdtk/Types.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include "n3rd/Layer.h"
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
    class MaxOverTimePoolingLayer : public Layer
    {

        int featureMapSz;
        int numFrames;
        std::vector<int> origin;
    public:




        /**
         * Default Ctor, used prior to rehydrating model from file
         */
        MaxOverTimePoolingLayer()
        {

        }


        MaxOverTimePoolingLayer(int numFeatureMaps) :
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


        sgdtk::Tensor& forward(const sgdtk::Tensor& x);

        // Since the output and input are the same for the max value, we can just apply the
        // max-pool value from the output

        sgdtk::Tensor& backward(sgdtk::Tensor& chainGrad, double y);

        std::string getType() const { return "MaxOverTimePoolingLayer"; }
    };

}

#endif
