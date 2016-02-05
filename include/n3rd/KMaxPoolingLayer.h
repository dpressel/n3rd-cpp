//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_KMAXPOOLINGLAYER_H__
#define __N3RD_CPP_KMAXPOOLINGLAYER_H__

#include <sgdtk/Types.h>
#include <sgdtk/DenseVectorN.h>
#include <cmath>
#include "n3rd/AbstractLayer.h"
#include "sgdtk/Tensor.h"
#include <algorithm>
namespace n3rd
{


    inline bool maxValue(sgdtk::Offset o1, sgdtk::Offset o2) { return o1.second > o2.second; }
    inline bool minIndex(sgdtk::Offset o1, sgdtk::Offset o2) { return o1.first < o2.first; }
    /**
     * K-max pooling, a generalization of max-pooling over time, where we take the top K values
     *
     * K-max pooling layer implements temporal max pooling, selecting up to K max features.  This is the approach used
     * in Kalchbrenner & Blunsom for their CNN sentence classification.  When K is 1, it simply becomes max-pooling over
     * time.
     *
     * The current implementation just uses builtin Java data structures, and isnt likely to be particularly optimal
     * and can likely be simplified quite a bit.
     *
     * @author dpressel
     */
    class KMaxPoolingLayer : public AbstractLayer<>
    {
        int k;
        int embeddingSz;
        int featureMapSz;
        int numFrames;
        std::vector<int> origin;
        std::vector<int> originDims;
    public:

        enum {DS_MIN = -1000000 };


        /**
         * Default Ctor, used prior to rehydrating model from file
         */
        KMaxPoolingLayer()
        {

        }

        /**
         * ructor for training
         * @param k The number of max values to use in each embedding
         * @param featureMapSz This is the number of feature maps
         * @param embedSz This is the embedding space, e.g, for some word2vec input, this might be something like 300
         */
        KMaxPoolingLayer(int kMax, int numFeatureMaps, int embedSz) :
                k(kMax), featureMapSz(numFeatureMaps), embeddingSz(embedSz)
        {
            output.resize({featureMapSz, embeddingSz, k});
            origin.resize(featureMapSz * embeddingSz * k);
        }

        int getK() const
        {
            return k;
        }

        void setK(int kMax)
        {
            k = kMax;
        }

        int getEmbeddingSz()
        {
            return embeddingSz;
        }

        void setEmbeddingSz(int embedSz)
        {
            embeddingSz = embedSz;
        }

        int getFeatureMapSz()
        {
            return featureMapSz;
        }

        void setFeatureMapSz(int numFeatureMaps)
        {
            featureMapSz = numFeatureMaps;
        }



        sgdtk::TensorI& forward(const sgdtk::TensorI& x);

        // Since the output and input are the same for the max value, we can just apply the
        // max-pool value from the output

        sgdtk::TensorI& backward(sgdtk::TensorI& chainGrad, double y);

        std::string getType() const { return "KMaxPoolingLayer"; }
    };

}

#endif
