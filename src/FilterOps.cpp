//
// Created by Daniel on 9/24/2015.
//

#include "n3rd/FilterOps.h"

using namespace sgdtk;
using namespace n3rd;

// Here we are applying the chain gradient from backprop (ygrad) as a cross-corr filter on the
// input layer.  This (of course) yields a weight gradient surface, which better be the same size as
// the weights themselves
void FilterOps::corr2Weights(const Tensor &x, const Tensor &ygrad, Tensor &weightGrads) {
    // x is then the input, and ygrad is the output, which is usually going to be smaller
    // x.dims[0] is #feature maps in the input
    // x.dims[1] is the #rows in input
    // x.dims[2] is the #cols in input

    const int nFeatureMapsInput = x.dims[0];
    const int xRows = x.dims[1];
    const int xCols = x.dims[2];


    // y is the deltas
    // y.dims[0] is the #feature maps in the output
    const int nFeatureMapsOutput = ygrad.dims[0];
    // y.dims[1] is the #rows in output
    const int yRows = ygrad.dims[1];

    // y.dims[2][ is the #cols in ouptut
    const int yCols = ygrad.dims[2];


    // The number of cubes is the output depth (# feature maps)
    weightGrads.resize({nFeatureMapsOutput, nFeatureMapsInput, xRows - yRows + 1, xCols - yCols + 1});

    const int kRows = weightGrads.dims[2];
    const int kCols = weightGrads.dims[3];

    // For each feature map
    for (int k = 0; k < nFeatureMapsInput; ++k) {
        // The weight gradient is the size of the kernel itself, which is a cube of size input depth x kh x kw
        //int okbase = k * kRows;

        for (int l = 0; l < nFeatureMapsOutput; ++l) {

            // For each input row
            for (int i = 0; i < kRows; ++i) {
                //int kbase = (okbase + i) * kCols;
                // For each input col
                for (int j = 0; j < kCols; ++j) {

                    int wAddr = ((l * nFeatureMapsInput + k) * kRows + i) * kCols + j;

                    // For input depth
                    double acc = 0;

                    // corr2!!
                    for (int m = 0; m < yRows; ++m) {
                        for (int n = 0; n < yCols; ++n) {
                            int xAddr = (k * xRows + i + m) * xCols + j + n;
                            //const int kh = m;//ygrad.h - m - 1;
                            //const int kw = n;//ygrad.w - n - 1;
                            int yAddr = (l * yRows + m) * yCols + n;
                            acc += x.d[xAddr] * ygrad.d[yAddr];
                        }
                    }

                    weightGrads.d[wAddr] = acc;
                }
            }
        }
    }

}

void FilterOps::corr1Weights(const Tensor &x, const Tensor &ygrad, Tensor &weightGrads) {
    // x is then the input, and ygrad is the output, which is usually going to be smaller
    // x.dims[0] is #feature maps in the input
    // x.dims[1] is the #rows in input
    // x.dims[2] is the #cols in input

    const int nFeatureMapsInput = x.dims[0];
    const int xRows = x.dims[2];
    const int embeddingSz = x.dims[1];


    // y is the deltas
    // y.dims[0] is the #feature maps in the output
    const int nFeatureMapsOutput = ygrad.dims[0];
    // y.dims[1] is the #rows in output
    const int yRows = ygrad.dims[2];

    const int kRows = weightGrads.dims[3];

    // For each feature map
    for (int k = 0; k < nFeatureMapsInput; ++k)
    {
        // The weight gradient is the size of the kernel itself, which is a cube of size input depth x kh x kw
        for (int l = 0; l < nFeatureMapsOutput; ++l)
        {
            for (int j = 0; j < embeddingSz; ++j)
            {
                // For each input row
                for (int i = 0; i < kRows; ++i)
                {
                    // For each input col, embeddingSz is also kCols essentially

                    int wAddr = ((l * nFeatureMapsInput + k) * embeddingSz + j) * kRows + i;
                    // For input depth
                    double acc = 0;

                    // corr2!!
                    for (int m = 0; m < yRows; ++m) {

                        int xAddr = (k * embeddingSz + j) * xRows + i + m;
                        int yAddr = (l * embeddingSz + j) * yRows + m;
                        acc += x.d[xAddr] * ygrad.d[yAddr];
                    }
                    weightGrads.d[wAddr] = acc;
                }
            }
        }
    }

}


void FilterOps::corr1(const Tensor &data, const Tensor &kernels, const std::vector<double> &biases, Tensor &output)
{
    const int iT = data.dims[2];
    const int embedSz = data.dims[1];
    const int nK = kernels.dims[0];
    const int kL = kernels.dims[1];
    const int kW = kernels.dims[3];
    const int oT = iT - kW + 1;
    //if (output.empty())
    //{
    //output.resize({nK, oT, embedSz});
    //}
    for (int k = 0, kbase = 0; k < nK; ++k, kbase += embedSz) {
        const double bias = biases.empty() ? 0.0 : biases[k];


        for (int j = 0; j < embedSz; ++j)
        {
            const int outAddr0 = (kbase + j) * oT;
            for (int i = 0; i < oT; ++i)
            {
                output[outAddr0 + i] = bias;
            }
            for (int l = 0, lbase = 0; l < kL; ++l, lbase += embedSz)
            {
                const int dataAddr0 = (lbase + j) * iT;
                const int kernAddr0 = ((k * kL + l) * embedSz + j) * kW;
                for (int i = 0; i < oT; ++i)
                {
                    const int outAddr = outAddr0 + i;
                    for (int m = 0; m < kW; ++m)
                    {
                        const int dataAddr = dataAddr0 + i + m;
                        const int kernAddr = kernAddr0 + m;
                        output[outAddr] += data.d[dataAddr] * kernels.d[kernAddr];
                    }
                }
            }
        }
    }
}


void FilterOps::conv1(const Tensor &data, const Tensor &kernels, const std::vector<double> &biases, Tensor &output)
{
    const int iT = data.dims[2];
    const int embedSz = data.dims[1];
    const int nK = kernels.dims[0];
    const int kL = kernels.dims[1];
    const int kW = kernels.dims[3];
    const int oT = iT - kW + 1;

    //output.resize({nK, oT, embedSz});

    for (int k = 0, kbase = 0; k < nK; ++k, kbase += embedSz)
    {
        const double bias = biases.empty() ? 0.0 : biases[k];

        for (int j = 0; j < embedSz; ++j)
        {
            const int outAddr0 = (kbase + j) * oT;

            for (int i = 0; i < oT; ++i)
            {
                output[outAddr0 + i] = bias;
            }
            for (int l = 0, lbase = 0; l < kL; ++l, lbase += embedSz)
            {
                const int dataAddr0 = (lbase + j) * iT;
                const int kernAddr0 = ((k * kL + l) * embedSz + j) * kW;

                for (int i = 0; i < oT; ++i)
                {
                    const int outAddr = outAddr0 + i;
                    for (int m = 0; m < kW; ++m)
                    {
                        const int dataAddr = dataAddr0 + i + m;
                        const int kernAddr = kernAddr0 + (kW - m - 1);
                        output[outAddr] += data.d[dataAddr] * kernels.d[kernAddr];
                    }
                }
            }
        }
    }
}





// In this case, we have several feature maps, each is a kernel of

void FilterOps::conv2(const Tensor &data, const Tensor &kernels, const std::vector<double> &biases, Tensor &output) {
    const int dH = data.dims[1];
    const int dW = data.dims[2];
    const int nK = kernels.dims[0];
    const int kL = kernels.dims[1];
    const int kH = kernels.dims[2];
    const int kW = kernels.dims[3];
    const int oH = dH - kH + 1;
    const int oW = dW - kW + 1;
    //if (output.empty())
    //{
    output.resize({nK, oH, oW});
    //}
    for (int k = 0; k < nK; ++k) {
        int kbase = k * kL;
        int obase = k * oH;

        const double bias = biases.empty() ? 0.0 : biases[k];
        for (int i = 0; i < oH; ++i) {
            int ibase = (obase + i) * oW;
            for (int j = 0; j < oW; ++j) {
                int outAddr = ibase + j;
                double acc = 0.;
                for (int l = 0; l < kL; ++l) {
                    for (int m = 0; m < kH; ++m) {
                        for (int n = 0; n < kW; ++n) {
                            int dataAddr = (l * dH + i + m) * dW + j + n;
                            int mh = kH - m - 1;
                            int nw = kW - n - 1;
                            int kernAddr = ((kbase + l) * kH + mh) * kW + nw;
                            acc += data.d[dataAddr] * kernels.d[kernAddr];
                        }
                    }
                }
                output.d[outAddr] = acc + bias;
            }
        }
    }

}

void FilterOps::corr2(const Tensor &data, const Tensor &kernels, const std::vector<double> &biases, Tensor &output) {
    const int dH = data.dims[1];
    const int dW = data.dims[2];
    const int nK = kernels.dims[0];
    const int kL = kernels.dims[1];
    const int kH = kernels.dims[2];
    const int kW = kernels.dims[3];
    const int oH = dH - kH + 1;
    const int oW = dW - kW + 1;
    //if (output.empty())
    //{
    output.resize({nK, oH, oW});
    //}
    for (int k = 0; k < nK; ++k) {
        int kbase = k * kL;
        int obase = k * oH;

        const double bias = biases.empty() ? 0.0 : biases[k];
        for (int i = 0; i < oH; ++i) {
            int ibase = (obase + i) * oW;
            for (int j = 0; j < oW; ++j) {
                int outAddr = ibase + j;
                double acc = 0.;
                for (int l = 0; l < kL; ++l) {
                    for (int m = 0; m < kH; ++m) {
                        for (int n = 0; n < kW; ++n) {
                            int dataAddr = (l * dH + i + m) * dW + j + n;
                            int kernAddr = ((kbase + l) * kH + m) * kW + n;
                            double d = data.d[dataAddr] * kernels.d[kernAddr];
                            acc += d;

                        }
                    }
                }
                output.d[outAddr] = acc + bias;

            }
        }
    }


}
