//
// Created by Daniel on 9/24/2015.
//

#ifndef __N3RD_CPP_TEMPORALCONVOLUTIONALLAYER_H__
#define __N3RD_CPP_TEMPORALCONVOLUTIONALLAYER_H__

#include <sgdtk/Tensor.h>
#include "n3rd/Layer.h"
#include "sgdtk/DenseVectorN.h"
#include <cmath>
#include <cstdlib>
#include "n3rd/FilterOps.h"

namespace n3rd
{

/**
 * Temporal convolution with support for feature maps AND preserving word embeddings
 * <p/>
 * This thing is different from the one in Torch.  In torch, the frame width is essentially a feature map
 * and the output is also.  This means that embeddings are not preserved between layers.
 * Assuming we want to preserve that locally, we would do this differently, making the embedding size 1,
 * and using the nK for the embeddingSz.  I believe this basically means that we can do everything Torch can, but also
 * we can do the Kalchbrenner/Blunsom thing as well. If you want to do the Torch approach, just pass and embeddingSz
 * of 1 and handle everything else outside
 */
    class TemporalConvolutionalLayer : public Layer
    {
    public:

        // Input is Number of frames x frame width (num feature maps)
        sgdtk::Tensor input;

        // Output is Number of frames x num feature maps


        ///Tensor gradsW;
        ///Tensor grads;
        ///int Current = 0;

        TemporalConvolutionalLayer()
        {

        }

        TemporalConvolutionalLayer(int nK, int kL, int kW, int embeddingSize);

        sgdtk::Tensor& forward(const sgdtk::Tensor& z);


        sgdtk::Tensor& backward(const sgdtk::Tensor& chainGrad, double y);

        std::string getType() const
        { return "TemporalConvolutionalLayer"; }
    };


}
#endif