#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>


#include "n3rd/TemporalConvolutionalLayer.h"
#include "n3rd/TemporalConvolutionalLayerCuBlas.h"
#include "n3rd/TemporalConvolutionalLayerBlas.h"
#include "n3rd/SpatialConvolutionalLayerBlas.h"
#include "n3rd/SpatialConvolutionalLayerCuBlas.h"
#include "n3rd/SpatialConvolutionalLayer.h"
using namespace sgdtk;
using namespace n3rd;

#include <cassert>

#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X

// embeddings ares
std::vector<double> DNOE =
        {
                1,3,5,7,9,11
        };

std::vector<double> KNOE = {
        1,2,3
};

std::vector<double> OFM1IFM1NOE = {22, 34, 46, 58};

std::vector<double> IFM2KNOE = {
        1, 2, 3,
        7, 8, 9
};

std::vector<double> OFM1IFM2NOE = {
        122., 182., 242., 302.
};

// This is what the unrolled matrix should look like (though its really in col-major form for BLAS)
std::vector<double> UNROLLED = {
        1,    3,    5,    2,    4,    6,
        3,    5,    7,    4,    6,    8,
        5,    7,    9,    6,    8,   10,
        7,    9,   11,    8,   10,   12

};

std::vector<double> IFM2DNOE = {
        1, 3, 5, 7, 9, 11,
        2, 4, 6, 8, 10, 12};


std::vector<double> IFM2OFM3KNOE = {

        -2, -1, 0, 1, 2, 3,

        4, 5, 6, 7, 8, 9,

        10, 11, 12, -13, -14, -15


};

std::vector<double> OFM3IFM2DNOE = {
        23., 29., 35., 41.,
        149., 227., 305., 383.,
        -69., -87., -105., -123.
};

double SQ_M_1000_1CHAN = 425.789216;
double SQ_M_W_1000_1CHAN = 2383.197216;

std::vector<double> OFM3IFM2G_1000_1CHAN = {
        -0.14, -0.057, 0.31499999999, 0.873, 1.090999999, 0.8219999, 1.963, 4.953, 9.072, 11.7359999, 9.293, 5.415

};


std::vector<double> K1222 = {
        .1, -.2, .3, -.4,
        -.7, .8, -.9, .11
};


std::vector<double> IFM243 = {
        1,    3,    5,    2,    4,    6,
        3,    5,    7,    4,    6,    8,
        5,    7,    9,    6,    8,   10,
        7,    9,   11,    8,   10,   12

};

std::vector<double> OFM132 = {
        -3.92, -5.7,
        -4.81, -6.59,
        -5.7, -7.48,
};


void unwrapInput(const sgdtk::Tensor& x, sgdtk::Tensor& unwrappedInput, int kW)
{
    int kL = x.dims[0];
    int numFrames = x.dims[2];
    const int oT = numFrames - kW + 1;
    unwrappedInput.resize({oT, kW * kL});
    int n = 0;


    for (int k = 0; k < kL; ++k)
    {

        for (int m = 0; m < kW; ++m)
        {
            for (int i = 0; i < oT; ++i)
            {

                int offset = k * numFrames + i + m;
                unwrappedInput[n] = x[offset];
                ++n;
            }
        }
    }
}

void wrapGrad(const sgdtk::Tensor& unwrapped, sgdtk::Tensor& grads, int kW)
{

    const int oT = unwrapped.dims[0];
    const int iT = oT + kW - 1;
    assert(iT == grads.dims[2]);
    const int kL = grads.dims[0];
    const int embedSz = grads.dims[1];
    assert(1 == embedSz);


    // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
    int n = 0;

    for (int k = 0; k < kL; ++k)
    {
        for (int m = 0; m < kW; ++m)
        {
            for (int i = 0; i < oT; ++i)
            {
                int offset = k * iT + i + m;
                // x(kL, iT, embedSz)
                grads[offset] += unwrapped[n];
                n++;
            }
        }
    }
}

void testWrapGrads()
{
    Tensor unwrapped(UNROLLED, {4,6});
    Tensor grads({2, 1, 6});
    wrapGrad(unwrapped, grads, 3);
    int sz = grads.size();

    CudaTensor dUnwrapped(unwrapped);
    CudaTensor dGrads(grads.dims);
    dGrads.constant(0);
    //assertEquals(grads.dims[2], unwrapped.dims[0]);
    //assertEquals(grads.dims[2] + 3 - 1, grads.dims[2]);
    n3rdgWrapGrad(dUnwrapped.d, dGrads.d, 2, 3, unwrapped.dims[0]);


    Tensor gradsGPU;

    dGrads.toCPU(gradsGPU);

    for (int i = 0; i < sz; ++i)
    {
        assertEqualsF(grads[i], gradsGPU[i], 1e-6);
    }

}

void testUnwrap()
{

    Tensor x(IFM2KNOE, {2,1,3});
    Tensor unwrappedInput;
    Tensor unwrappedGPU;

    CudaTensor dX(x);


    unwrapInput(x, unwrappedInput, 3);
    CudaTensor dUnwrapped(unwrappedInput.dims);

    n3rdgUnwrapInput(dX.d, dUnwrapped.d, 2, 3, 3);

    dUnwrapped.toCPU(unwrappedGPU);

    int gsz = unwrappedGPU.size();
    int sz = unwrappedInput.size();

    assertEquals(sz, gsz);
    for  (int i = 0 ; i < sz; ++i)
    {
        assertEqualsF(unwrappedInput[i], unwrappedGPU[i], 1e-6);
    }

    Tensor x2(IFM2DNOE, {2,1,6});
    dX = x2;

    unwrapInput(x2, unwrappedInput, 3);
    dUnwrapped.resize(unwrappedInput.dims);


    n3rdgUnwrapInput(dX.d, dUnwrapped.d, 2, 3, 6);

    dUnwrapped.toCPU(unwrappedGPU);

    gsz = unwrappedGPU.size();
    sz = unwrappedInput.size();

    assertEquals(sz, gsz);
    for  (int i = 0 ; i < sz; ++i)
    {
        assertEqualsF(unwrappedInput[i], unwrappedGPU[i], 1e-6);
    }
}


void testForward2D() throw(Exception)
{
    int nK = 1;
    int kH = 2;
    int kW = 2;
    std::vector<int> inputDims = {2,4,3};
    auto* l = new SpatialConvolutionalLayer(nK, kH, kW, inputDims);
    auto* lb = new SpatialConvolutionalLayerBlas(nK, kH, kW, inputDims);

    auto& weights = (Tensor&)l->getParams();
    auto& weightsb = (Tensor&)lb->getParams();
    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < K1222.size(); ++i)
    {
        weights[i] = K1222[i];
        weightsb[i] = K1222[i];
    }


    Tensor d(IFM243, inputDims);
    Tensor& output = (Tensor&) l->forward(d);

    assertEquals(output.dims[0], 1);
    assertEquals(output.dims[1], 3);
    assertEquals(output.dims[2], 2);

    Tensor& outputb = (Tensor&) lb->forward(d);


    for (int i = 0; i < output.size(); ++i)
    {
        assertEqualsF(OFM132[i], output[i], 1e-3);
        assertEqualsF(OFM132[i], outputb[i], 1e-3);
    }

}


void testForward2DGPU() throw(Exception)
{
    int nK = 1;
    int kH = 2;
    int kW = 2;
    int kL = 2;
    const int ncols = 3;
    const int nrows = 4;

    std::vector<int> inputDims = {kL, nrows, ncols};

    auto* l = new SpatialConvolutionalLayerCuBlas(nK, kH, kW, inputDims);

    auto& gpuWeights = (CudaTensor&)l->getParams();

    Tensor weights(gpuWeights.dims);


    int n = 0;

    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < K1222.size(); ++i)
    {
        weights[i] = K1222[i];
    }

    Tensor over;

    gpuWeights.toCPU(over);

    gpuWeights.fromCPU(weights, false);
    Tensor d(IFM243, inputDims);

    CudaTensor dD(d);

    Tensor z;
    dD.toCPU(z);
    auto& dOutput = (CudaTensor&)l->forward(dD);

    Tensor output;
    dOutput.toCPU(output);

    assertEquals(output.dims[0], 1);
    assertEquals(output.dims[1], 3);
    assertEquals(output.dims[2], 2);

    for (int i = 0; i < output.size(); ++i)
    {
        assertEqualsF(OFM132[i], output[i], 1e-3);
    }

}


void testBackward2D() throw(Exception)
{
    int nK = 1;
    int kH = 2;
    int kW = 2;
    std::vector<int> inputDims = {2,4,3};
    auto* l = new SpatialConvolutionalLayer(nK, kH, kW, inputDims);
    auto* lb = new SpatialConvolutionalLayerBlas(nK, kH, kW, inputDims);

    auto& weights = (Tensor&)l->getParams();
    auto& weightsb = (Tensor&)lb->getParams();
    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < K1222.size(); ++i)
    {
        weights[i] = K1222[i];
        weightsb[i] = K1222[i];
    }


    Tensor d(IFM243, inputDims);
    Tensor& output = (Tensor&) l->forward(d);
    Tensor& outputb = (Tensor&) lb->forward(d);

    output.scale(1/1000.);
    outputb.scale(1/1000.);

    Tensor& bwd = (Tensor&)l->backward(output, 0);

    Tensor& bwdb = (Tensor&)l->backward(outputb, 0);

    for (int i = 0; i < bwd.size(); ++i)
    {
        assertEqualsF(bwd[i], bwdb[i], 1e-6);
    }
}


void testBackward2DGPU() throw(Exception)
{
    int nK = 1;
    int kH = 2;
    int kW = 2;
    std::vector<int> inputDims = {2,4,3};
    auto* l = new SpatialConvolutionalLayer(nK, kH, kW, inputDims);
    auto* lb = new SpatialConvolutionalLayerCuBlas(nK, kH, kW, inputDims);

    auto& weights = (Tensor&)l->getParams();
    auto& dWeightsb = (CudaTensor&)lb->getParams();

    Tensor weightsb(dWeightsb.dims);


    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < K1222.size(); ++i)
    {
        weights[i] = K1222[i];
        weightsb[i] = K1222[i];
    }

    dWeightsb.fromCPU(weightsb);

    Tensor d(IFM243, inputDims);
    CudaTensor dD(d);
    Tensor& output = (Tensor&) l->forward(d);
    Tensor& dOutputb = (Tensor&) lb->forward(dD);

    output.scale(1/1000.);
    dOutputb.scale(1/1000.);

    Tensor& bwd = (Tensor&)l->backward(output, 0);

    CudaTensor& dBwdb = (CudaTensor&)l->backward(dOutputb, 0);

    Tensor bwdb(dBwdb.dims);
    dBwdb.toCPU(bwdb);
    for (int i = 0; i < bwd.size(); ++i)
    {
        assertEqualsF(bwd[i], bwdb[i], 1e-6);
    }
}



void testForwardWordVecAsInChannels() throw(Exception)
{
    auto* l = new TemporalConvolutionalLayerBlas(1, 1, 3);

    auto& weights = (Tensor&)l->getParams();
    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < KNOE.size(); ++i)
    {
        weights[i] = KNOE[i];
    }
    Tensor d(DNOE, {1, 1, 6});
    Tensor& output = (Tensor&) l->forward(d);

    assertEquals(output.size(), OFM1IFM1NOE.size());

    for (int i = 0; i < OFM1IFM1NOE.size(); ++i)
    {
        assertEqualsF(output[i], OFM1IFM1NOE[i], 1e-6);
    }

}


void testForwardWordVecAsInChannelsGPU() throw(Exception)
{
    auto* l = new TemporalConvolutionalLayerCuBlas(1, 1, 3);

    auto& gpuWeights = (CudaTensor&)l->getParams();

    Tensor weights(gpuWeights.dims);

    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < KNOE.size(); ++i)
    {
        weights[i] = KNOE[i];
    }
    gpuWeights.fromCPU(weights, false);

    Tensor rtWeights;
    gpuWeights.toCPU(rtWeights);
    assertEquals(rtWeights.size(), weights.size());
    for (int i = 0; i < rtWeights.size(); ++i)
    {
        assertEqualsF(rtWeights[i], weights[i], 1e-6);
    };


    Tensor d(DNOE, {1, 1, 6});
    CudaTensor dD(d);
    auto& dOutput = (CudaTensor&) l->forward(dD);

    Tensor output;
    dOutput.toCPU(output);
    assertEquals(output.size(), OFM1IFM1NOE.size());

    for (int i = 0; i < OFM1IFM1NOE.size(); ++i)
    {
        assertEqualsF(output[i], OFM1IFM1NOE[i], 1e-6);
    }

}

void testForward2to1WordVecAsInChannels() throw(Exception)
{

    auto* l = new TemporalConvolutionalLayerBlas(1, 2, 3);

    Tensor& weights = (Tensor&)l->getParams();

    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < IFM2KNOE.size(); ++i)
    {
        weights[i] = IFM2KNOE[i];
    }

    Tensor d(IFM2DNOE, {2, 1, 6});
    auto& output = (Tensor&)l->forward(d);

    for (int i = 0; i < OFM1IFM2NOE.size(); ++i)
    {
        assertEqualsF(output[i], OFM1IFM2NOE[i], 1e-6);
    }

}




void testForward2to1WordVecAsInChannelsGPU() throw(Exception)
{

    auto* l = new TemporalConvolutionalLayerCuBlas(1, 2, 3);

    auto& gpuWeights = (CudaTensor&)l->getParams();

    Tensor weights(gpuWeights.dims);


    // Because there are no output feature maps, memory is same for col and row major
    for (int i = 0; i < IFM2KNOE.size(); ++i)
    {
        weights[i] = IFM2KNOE[i];
    }
    gpuWeights.fromCPU(weights, false);

    Tensor d(IFM2DNOE, {2, 1, 6});
    CudaTensor dD(d);
    auto& dOutput = (CudaTensor&)l->forward(dD);

    Tensor output;
    dOutput.toCPU(output);

    for (int i = 0; i < OFM1IFM2NOE.size(); ++i)
    {
        assertEqualsF(output[i], OFM1IFM2NOE[i], 1e-6);
    }

}



void testForward2to3WordVecAsInChannels() throw(Exception)
{

    auto* l = new TemporalConvolutionalLayerBlas(3, 2, 3);

    auto& w = (Tensor&)l->getParams();
    assertEquals(w.size(), IFM2OFM3KNOE.size());

    int nrows = 6;
    int ncols = 3;
    int n = 0;
    for (int j = 0; j < ncols; ++j)
    {
        for (int i = 0; i < nrows; ++i)
        {
            w[j * nrows + i] = IFM2OFM3KNOE[n++];
        }
    }

    Tensor d(IFM2DNOE, {2, 1, 6});
    auto& output = (Tensor&)l->forward(d);

    assertEquals(output.size(), OFM3IFM2DNOE.size());
    for (int i = 0; i < OFM3IFM2DNOE.size(); ++i)
    {
        assertEqualsF(output[i], OFM3IFM2DNOE[i], 1e-6);
    }

}




void testForward2to3WordVecAsInChannelsGPU() throw(Exception)
{

    auto* l = new TemporalConvolutionalLayerCuBlas(3, 2, 3);

    auto& gpuWeights = (CudaTensor&)l->getParams();
    assertEquals(gpuWeights.size(), IFM2OFM3KNOE.size());

    int nrows = 6;
    int ncols = 3;
    int n = 0;

    Tensor w(gpuWeights.dims);

    for (int j = 0; j < ncols; ++j)
    {
        for (int i = 0; i < nrows; ++i)
        {
            w[j * nrows + i] = IFM2OFM3KNOE[n++];
        }
    }

    gpuWeights.fromCPU(w, false);

    Tensor d(IFM2DNOE, {2, 1, 6});
    CudaTensor dD(d);

    auto& dOutput = (CudaTensor&)l->forward(dD);

    Tensor output;
    dOutput.toCPU(output);

    assertEquals(output.size(), OFM3IFM2DNOE.size());
    for (int i = 0; i < OFM3IFM2DNOE.size(); ++i)
    {
        assertEqualsF(output[i], OFM3IFM2DNOE[i], 1e-6);
    }

}



void testBackward2to3WordVecAsInChannels() throw(Exception)
{
    auto* l = new TemporalConvolutionalLayerBlas(3, 2, 3);
    auto& w = (Tensor&)l->getParams();
    assertEquals(w.size(), IFM2OFM3KNOE.size());

    int nrows = 6;
    int ncols = 3;

    int n = 0;
    for (int j = 0; j < ncols; ++j)
    {
        for (int i = 0; i < nrows; ++i)
        {
            w[j * nrows + i] = IFM2OFM3KNOE[n++];
        }
    }

    Tensor d(IFM2DNOE, {2, 1, 6});
    auto& output = (Tensor&)l->forward(d);

    output.scale(1 / 1000.);

    auto& grads = (Tensor&)l->backward(output, 0);

    auto& gw = (Tensor&)l->getParamGrads();

    // Are gradients right?
    double acc = 0.;
    // Are weights right after gradients applied?
    double accW = 0.;
    for (int i = 0, gsz = gw.size(); i < gsz; ++i)
    {
        acc += gw[i] * gw[i];
        //weights.addi(i, gw.get(i));
        auto wU = w[i] + gw[i];
        accW += wU * wU;
        gw[i] = 0;
    }
    assertEqualsF(SQ_M_1000_1CHAN, acc, 1e-6);
    assertEqualsF(SQ_M_W_1000_1CHAN, accW, 1e-6);
    assertEquals(grads.size(), OFM3IFM2G_1000_1CHAN.size());

    for (int i = 0; i < OFM3IFM2G_1000_1CHAN.size(); ++i)
    {
        assertEqualsF(OFM3IFM2G_1000_1CHAN[i], grads[i], 1e-3);
    }
}


void testBackward2to3WordVecAsInChannelsGPU() throw(Exception)
{
    auto* l = new TemporalConvolutionalLayerCuBlas(3, 2, 3);
    auto& gpuWeights = (CudaTensor&)l->getParams();
    Tensor w(gpuWeights.dims);
    assertEquals(w.size(), IFM2OFM3KNOE.size());

    int nrows = 6;
    int ncols = 3;

    int n = 0;
    for (int j = 0; j < ncols; ++j)
    {
        for (int i = 0; i < nrows; ++i)
        {
            w[j * nrows + i] = IFM2OFM3KNOE[n++];
        }
    }

    gpuWeights.fromCPU(w, false);

    Tensor d(IFM2DNOE, {2, 1, 6});
    CudaTensor dD(d);
    auto& dOutput = (CudaTensor&)l->forward(dD);

    dOutput.scale(1 / 1000.);

    auto& dGrads = (CudaTensor&)l->backward(dOutput, 0);

    Tensor grads;
    dGrads.toCPU(grads);

    auto& dGw = (CudaTensor&)l->getParamGrads();

    Tensor gw;
    dGw.toCPU(gw);

    // Are gradients right?
    double acc = 0.;
    // Are weights right after gradients applied?
    double accW = 0.;
    for (int i = 0, gsz = gw.size(); i < gsz; ++i)
    {
        acc += gw[i] * gw[i];
        //weights.addi(i, gw.get(i));
        auto wU = w[i] + gw[i];
        accW += wU * wU;
        gw[i] = 0;
    }
    assertEqualsF(SQ_M_1000_1CHAN, acc, 1e-6);
    assertEqualsF(SQ_M_W_1000_1CHAN, accW, 1e-6);
    assertEquals(grads.size(), OFM3IFM2G_1000_1CHAN.size());

    for (int i = 0; i < OFM3IFM2G_1000_1CHAN.size(); ++i)
    {
        assertEqualsF(OFM3IFM2G_1000_1CHAN[i], grads[i], 1e-3);
    }
}


int main(int argc, char **argv)
{

    try
    {
        initCuBlas();
        EVAL(testForward2D());
        EVAL(testForward2DGPU());
        EVAL(testBackward2D());
        EVAL(testBackward2DGPU());

        EVAL(testForwardWordVecAsInChannels());
        EVAL(testForward2to1WordVecAsInChannels());
        EVAL(testForward2to3WordVecAsInChannels());
        EVAL(testBackward2to3WordVecAsInChannels());

        EVAL(testForwardWordVecAsInChannelsGPU());
        EVAL(testForward2to1WordVecAsInChannelsGPU());
        EVAL(testForward2to3WordVecAsInChannelsGPU());
        EVAL(testBackward2to3WordVecAsInChannelsGPU());

        EVAL(testUnwrap());
        EVAL(testWrapGrads());
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }
    doneCuBlas();
    return 0;
}