#include "sgdtk/sgdtk.h"

#include <iostream>
#include <map>
#include "n3rd/FullyConnectedLayer.h"
#include "n3rd/FullyConnectedLayerCuBlas.h"

using namespace sgdtk;
using namespace n3rd;

#include <cassert>

#define assertEquals(X, Y) assert(X == Y)

#define assertEqualsF(X, Y, EPS) assert(std::abs(X - Y) < EPS)

#define EVAL(X) std::cout << #X << std::endl; X
std::vector<double> K = {1, 2, 3, 4, 5, 6, 7, 8};
const int M = 2;
const int N = 4;
std::vector<double> V_4 = {0.4, 0.3, 0.2, 0.1};
std::vector<double> V_2 = {2, 6};
std::vector<double> D4_X = {32.0, 40.0, 48.0, 56.0};
std::vector<double> W_G = {0.8, 0.6, 0.4, 0.2,
                           2.4, 1.8, 1.2, 0.6};

std::vector<double> W_G2 = {0.8, 2.4, 0.6, 1.8, 0.4, 1.2, 0.2, 0.6};
std::vector<double> D2_X = {3, 4};
void testForward() throw(Exception)
{
    auto *dfc = new FullyConnectedLayerCuBlas(2, 4);
    auto *fc = new FullyConnectedLayer(2, 4);

    auto &b = (Tensor&)fc->getBiasParams();
    auto &w = (Tensor&)fc->getParams();
    auto &dB = (CudaTensor&)dfc->getBiasParams();
    auto &dW = (CudaTensor&)dfc->getParams();

    Tensor nw(w.dims);
    Tensor nb(b.dims);

    assertEquals(K.size(), w.size());


    int n = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int idx = i * N + j;
            int cIdx = j * M + i;
            w[idx] = K[n];
            nw[cIdx] = K[n];
            n++;
        }
    }

    for (int i = 0; i < b.size(); ++i)
    {
        b[i] = nb[i] = 0.;
    }

    dB.fromCPU(nb);
    dW.fromCPU(nw);

    Tensor d(V_4, {4});
    CudaTensor dD(d);

    auto &o = (Tensor&)fc->forward(d);

    auto &dO = (CudaTensor&)dfc->forward(dD);
    assertEquals(o.size(), V_2.size());
    assertEquals(dO.size(), V_2.size());

    Tensor no;
    dO.toCPU(no);

    for (int i = 0; i < V_2.size(); ++i)
    {
        std::cout << no[i] << std::endl;
        assertEquals(no[i], V_2[i]);
        assertEquals(o[i], V_2[i]);
    }
    delete fc;

}


void testBackward4to2() throw(Exception)
{
    auto *fc = new FullyConnectedLayer(2, 4);
    auto &w = (Tensor&)fc->getParams();
    auto &b = (Tensor&)fc->getBiasParams();


    auto *dfc = new FullyConnectedLayerCuBlas(2, 4);
    auto &dB = (CudaTensor&)dfc->getBiasParams();
    auto &dW = (CudaTensor&)dfc->getParams();

    Tensor nw(w.dims);
    Tensor nb(b.dims);
    assertEquals(K.size(), w.size());


    int n = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int idx = i * N + j;
            int cIdx = j * M + i;
            w[idx] = K[n];
            nw[cIdx] = K[n];
            n++;
        }
    }


    for (int i = 0; i < b.size(); ++i)
    {
        b[i] = nb[i] = 0.;
    }


    dB.fromCPU(nb);
    dW.fromCPU(nw);

    assertEquals(K.size(), w.size());


    Tensor error(V_2, {2});
    Tensor d(V_4, {4});

    CudaTensor dD(d);
    fc->forward(d);

    auto &v = (Tensor&)fc->backward(error, 0.);


    dfc->forward(dD);

    CudaTensor dError(error);
    auto &dV = (CudaTensor&)dfc->backward(dError, 0.);

    Tensor nv;
    dV.toCPU(nv);
    std::cout << "Deltas" << std::endl;
    assertEquals(D4_X.size(), v.size());
    for (int i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
        assertEquals(D4_X[i], nv[i]);
        assertEquals(v[i], v[i]);

    }
    std::cout << std::endl << std::endl;


    auto &gw = (Tensor&)fc->getParamGrads();
    auto &dGw = (CudaTensor&)dfc->getParamGrads();
    assertEquals(W_G.size(), gw.size());
    assertEquals(gw.size(), dGw.size());

    Tensor ngw;
    dGw.toCPU(ngw);

    std::cout << "Wg" << std::endl;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j) {

            assertEqualsF(ngw[j * M + i], gw[i * N + j], 1e-6);
            std::cout << gw[i] << " ";
        }
    }
    std::cout << std::endl;
    delete fc;
}
void testBackward2to4() throw(Exception)
{
    auto* fc = new FullyConnectedLayer(4, 2);
    auto& w = (Tensor&)fc->getParams();
    auto& b = (Tensor&)fc->getBiasParams();
    assertEquals(K.size(), w.size());
    auto* dfc = new FullyConnectedLayerCuBlas(4, 2);
    auto& dW = (CudaTensor&)dfc->getParams();
    auto& dB = (CudaTensor&)dfc->getBiasParams();


    Tensor nw(w.dims);
    Tensor nb(b.dims);

    assertEquals(K.size(), nw.size());

    int n = 0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            int idx = i * M + j;
            int cIdx = j * N + i;
            nw.d[cIdx] = K[n];
            w.d[idx] = K[n];
            n++;
        }
    }

    for (int i = 0; i < b.size(); ++i)
    {
        b[i] = nb[i] = 0.;
    }


    dB.fromCPU(nb);
    dW.fromCPU(nw);


    Tensor error(V_4, {V_4.size()});
    Tensor d(V_2, {V_2.size()});
    fc->forward(d);
    auto& v = (Tensor&)fc->backward(error, 0.);

    CudaTensor dD(d);
        //d = new DenseVectorN(V_2);
    dfc->forward(dD);
    CudaTensor dError(error);
    auto& dNv = (CudaTensor&)dfc->backward(dError, 0.);

    Tensor nv;
    dNv.toCPU(nv);
    for (int i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
        assertEqualsF(v[i], nv[i], 1e-6);
    }
    std::cout << std::endl;

    auto& gw = (Tensor&)fc->getParamGrads();
    auto& dGw = (CudaTensor&)dfc->getParamGrads();

    assertEquals(gw.size(), dGw.size());

    Tensor ngw;
    dGw.toCPU(ngw);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            assertEqualsF(ngw[j * N + i], gw[i * M + j], 1e-6);
            std::cout << gw[i * M + j] << " ";
        }
        std::cout << std::endl;

    }
    delete fc;
}

int main(int argc, char **argv)
{

    try
    {

        initCuBlas();
        EVAL(testForward());
        EVAL(testBackward4to2());
        EVAL(testBackward2to4());
        std::cout << "SUCCESS!" << std::endl;
    }
    catch (Exception &ex)
    {
        std::cout << "FAILURE: " << ex.getMessage() << std::endl;
    }
    doneCuBlas();

    return 0;
}