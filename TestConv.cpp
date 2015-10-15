#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>

#include "n3rd/TemporalConvolutionalLayer.h"

using namespace sgdtk;
using namespace n3rd;

#include <cassert>

#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X

// embeddings ares
std::vector<double> D =
        {
                1, 2,
                3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12
        };

std::vector<double> K = {1, 4,
                         2, 5,
                         3, 6};


std::vector<double> OFM1IFM1 = {22, 64, 34, 94, 46, 124, 58, 154};

std::vector<double> IFM2K = {1, 7,
                             2, 8,
                             3, 9,
                             4, 10,
                             5, 11,
                             6, 12};

std::vector<double> OFM1IFM2 = {78.5, 252.5,
                                120.5, 366.5,
                                162.5, 480.5,
                                204.5, 594.5};

std::vector<double> IFM2D = {1, 2,
                             3, 4,
                             5, 6,
                             7, 8,
                             9, 10,
                             11, 12,
                             1.5, 2.5,
                             3.5, 4.5,
                             5.5, 6.5,
                             7.5, 8.5,
                             9.5, 10.5,
                             11.5, 12.5};

std::vector<double> IFM2OFM3K = {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12,
                                 1.5, 7.5, 2.5, 8.5, 3.5, 9.5, 4.5, 10.5, 5.5, 11.5, 6.5, 12.5,
                                 1.2, 7.2, 2.2, 8.2, 3.2, 9.2, 4.2, 10.2, 5.2, 11.2, 6.2, 12.2};

std::vector<double> OFM3IFM2D = {78.5,
                                 252.5,
                                 120.5,
                                 366.5,
                                 162.5,
                                 480.5,
                                 204.5,
                                 594.5,
                                 88.25,
                                 265.25,
                                 136.25,
                                 385.25,
                                 184.25,
                                 505.25,
                                 232.25,
                                 625.25,
                                 82.4,
                                 257.6,
                                 126.8,
                                 374.0,
                                 171.2,
                                 490.4,
                                 215.6,
                                 606.8};

std::vector<double> OFM3IFM2G_1000 = {
        0.30975499999,
        5.611595,
        1.03594,
        14.53462,
        2.312955,
        27.119474999,
        3.217995,
        35.778915,
        3.1441600000,
        28.682439999,
        2.116295,
        16.872934999,
        1.057205,
        7.937645,
        2.93404,
        20.23792,
        5.7649050000,
        37.251224999,
        7.8795450000,
        49.064265,
        6.6550600000,
        38.59054,
        4.0733450000,
        22.352585
};
double SQ_M_1000 = 3886.2073516200003;
double SQ_M_W_1000 = 11477.130271620003;

void testForward() throw(Exception)
{
    TemporalConvolutionalLayer *l = new TemporalConvolutionalLayer(1, 1, 3, 2);
    auto &w = l->getParams();
    for (int i = 0; i < K.size(); ++i)
    {
        w[i] = K[i];
    }
    Tensor d(D, {1, 6, 2});
    Tensor &output = l->forward(d);

    assertEquals(output.size(), OFM1IFM1.size());

    for (int i = 0; i < OFM1IFM1.size(); ++i)
    {
        assertEquals(output[i], OFM1IFM1[i]);
    }

}

void testForward2to1() throw(Exception)
{

    TemporalConvolutionalLayer *l = new TemporalConvolutionalLayer(1, 2, 3, 2);

    Tensor &weights = l->getParams();
    for (int i = 0; i < IFM2K.size(); ++i)
    {
        weights.d[i] = IFM2K[i];
    }


    Tensor d(IFM2D, {2, 6, 2});
    Tensor &output = l->forward(d);

    assertEquals(OFM1IFM2.size(), output.size());
    for (int i = 0; i < OFM1IFM2.size(); ++i)
    {
        assertEquals(output[i], OFM1IFM2[i]);
    }

}

void testForward2to3() throw(Exception)
{

    TemporalConvolutionalLayer *l = new TemporalConvolutionalLayer(3, 2, 3, 2);

    Tensor &weights = l->getParams();
    for (int i = 0; i < IFM2OFM3K.size(); ++i)
    {
        weights.d[i] = IFM2OFM3K[i];
    }

    Tensor d(IFM2D, {2, 6, 2});
    Tensor &output = l->forward(d);

    assertEquals(output.size(), OFM3IFM2D.size());
    for (int i = 0; i < output.size(); ++i)
    {
        assertEqualsF(output[i], OFM3IFM2D[i], 1e-6);
    }
    //printRowMajor(((DenseVectorN) output).getX(), 3, 4, 2);
}

void testBackward2to3() throw(Exception)
{
    TemporalConvolutionalLayer* l = new TemporalConvolutionalLayer(3, 2, 3, 2);

    auto& weights = l->getParams();
    for (int i = 0; i < IFM2OFM3K.size(); ++i)
    {
        weights.d[i] = IFM2OFM3K[i];
    }

    Tensor d(IFM2D, {2, 6, 2});
    auto& ograd = l->forward(d);

    for (int i = 0; i < ograd.size(); ++i)
    {
        ograd[i] /= 1000.;
    }
    auto& deltas = l->backward(ograd, 0);

    auto& gw = l->getParamGrads();

    // Are gradients right?
    double acc = 0.;
    // Are weights right after gradients applied?
    double accW = 0.;
    for (int i = 0; i < gw.size(); ++i)
    {
        acc += gw.d[i] * gw.d[i];
        weights.d[i] += gw.d[i];
        accW += weights.d[i] * weights.d[i];
        gw.d[i] = 0;
    }
    assertEqualsF(SQ_M_1000, acc, 1e-6);
    assertEqualsF(SQ_M_W_1000, accW, 1e-6);


    for (int i = 0; i < deltas.size(); ++i)
    {
        assertEqualsF(OFM3IFM2G_1000[i], deltas[i], 1e-6);
    }
}

int main(int argc, char **argv)
{

    try
    {

        EVAL(testForward());
        EVAL(testForward2to1());
        EVAL(testForward2to3());
        EVAL(testBackward2to3());
        ///      EVAL(testBackward4to2());
        ///   EVAL(testBackward2to4());
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }

    return 0;
}