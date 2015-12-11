#include "sgdtk/Tensor.h"

using namespace sgdtk;


double Tensor::mag() const
{
#ifdef USE_BLAS
    const double* v1 = &d[0];
    return cblas_ddot(d.size(), v1, 1, v1, 1);
#else
    auto acc = 0.0;
    for (auto v : d)
    {
        acc += v * v;
    }
    return acc;
#endif
}

void Tensor::scale(double scalar)
{
#ifdef USE_BLAS
    double* v1 = &d[0];
    cblas_dscal(d.size(), scalar, v1, 1);
#else
    for (int i = 0, sz = d.size(); i < sz; ++i)
    {
        d[i] *= scalar;
    }
#endif
}

//
//Tensor *embed(const Tensor *tensor, int h, int w)
//{
//    return embed(tensor, 0, h, w);
//}
