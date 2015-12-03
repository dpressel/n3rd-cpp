#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>


#include "n3rd/TemporalConvolutionalLayer.h"
//#include "n3rd/TemporalConvolutionalLayerCuDNN.h"
#include "n3rd/TemporalConvolutionalLayerCuBlas.h"
using namespace sgdtk;
using namespace n3rd;

#include <cassert>

#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X

// embeddings ares
std::vector<double> D =
        {
                1,3,5,7,9,11,
                2,4,6,8,10,12
        };

std::vector<double> K = {
        1,2,3,
        4,5,6
};

std::vector<double> O = {
        161, 137, 124, 122, 131, 151, 182
};
std::vector<double> OFM1IFM1 = {22,34,46,58,64,94,124,154};

std::vector<double> IFM2K = {  1,2,3,
                               7,8,9,
                               4,5,6,
                               10,11,12};

std::vector<double> OFM1IFM2 = { 78.5, 120.5, 162.5, 204.5,
                                 252.5, 366.5, 480.5, 594.5};

std::vector<double> IFM2D = {1,3,5,7,9,11,
                             2,4,6,8,10,12,
                             1.5,3.5,5.5,7.5,9.5,11.5,
                             2.5,4.5,6.5,8.5,10.5,12.5};

std::vector<double> IFM2OFM3K = { 1,2,3,
                                  7,8,9,

                                  4,5,6,
                                  10,11,12,



                                  1.5,2.5,3.5,
                                  7.5,8.5,9.5,

                                  4.5,5.5,6.5,
                                  10.5,11.5,12.5,



                                  1.2,2.2,3.2,
                                  7.2,8.2,9.2,

                                  4.2,5.2,6.2,
                                  10.2,11.2,12.2};

std::vector<double> OFM3IFM2D = { 78.5,120.5,162.5,204.5,
                                  252.5,366.5,480.5,594.5,

                                  88.25,136.25,184.25,232.25,
                                  265.25,385.25,505.25,625.25,

                                  82.4,126.8,171.2,215.6,
                                  257.6,374.0,490.4,606.8};

std::vector<double> OFM3IFM2G_1000 = {
        0.30975499999,1.03594,2.312955,3.217995,3.14416,2.116295,
        5.611595,14.53462,27.119474999,35.778915,28.682439999,16.872934999,

        1.057205,2.93404,5.764905,7.879545,6.65506,4.073345,
        7.937645,20.23792, 37.251224999,49.064265,38.59054,22.352585

};
double SQ_M_1000 = 3886.2073516200003;
double SQ_M_W_1000 = 11477.130271620003;

void testForward() throw(Exception)
{
    //TemporalConvolutionalLayerCuDNN *l = new TemporalConvolutionalLayerCuDNN(gHandle, 1, 1, 6, 1);
    TemporalConvolutionalLayerCuBlas* l = new TemporalConvolutionalLayerCuBlas(1,1,6);
    auto &w = l->getParams();
    for (int i = 0; i < K.size(); ++i)
    {
        w[i] = K[i];
    }
    //l->copyWeights();

    auto& t = l->getParams();

    Tensor d(D, {1, 1, 12});
    Tensor &output = l->forward(d);

    assertEquals(output.size(), O.size());

    for (int i = 0; i < output.size(); ++i)
    {
        std::cout << output[i] << std::endl;
        assertEquals(output[i], O[i]);
    }

}



int main(int argc, char **argv)
{

    try
    {
        initCuBlas();
        EVAL(testForward());
        //EVAL(testForward2to1());
        ////EVAL(testForward2to3());
        //EVAL(testBackward2to3());
        ///      EVAL(testBackward4to2());
        ///   EVAL(testBackward2to4());
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }
    doneCuBlas();
    return 0;
}