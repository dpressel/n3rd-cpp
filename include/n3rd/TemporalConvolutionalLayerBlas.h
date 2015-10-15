#ifndef __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERBLAS_H__
#define __N3RD_CPP_TEMPORALCONVOLUTIONALLAYERBLAS_H__

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
    class TemporalConvolutionalLayerBlas : public Layer
    {
        void reorderOutput(sgdtk::Tensor& unwrapped, int nK, int embedSz);
        void unwrapGrad(const sgdtk::Tensor& chainGrad, int nK, int embedSz, sgdtk::Tensor& unwrapped);
        void unwrapX(const sgdtk::Tensor& x, int kW, sgdtk::Tensor& unwrapped);
        void wrapX(const sgdtk::Tensor& unwrapped, sgdtk::Tensor& grads, int kW);
        int nK;
        int kL;
        int kW;
        int embedSz;
        int numFrames;
        sgdtk::Tensor input;
        sgdtk::Tensor unwrappedInput;
    public:


        // Input is Number of frames x frame width (num feature maps)
        sgdtk::Tensor z;

        // Output is Number of frames x num feature maps


        ///Tensor gradsW;
        ///Tensor grads;
        //int Current = 0;

        TemporalConvolutionalLayerBlas()
        {

        }

        TemporalConvolutionalLayerBlas(int nK, int kL, int kW, int embeddingSize);
        ~TemporalConvolutionalLayerBlas()
        {

        }


        sgdtk::Tensor& forward(const sgdtk::Tensor& input);


        sgdtk::Tensor& backward(const sgdtk::Tensor& chainGrad, double y);

        std::string getType() const
        { return "TemporalConvolutionalLayerBlas"; }
    };


}
#endif