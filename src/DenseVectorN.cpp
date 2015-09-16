#include "sgdtk/DenseVectorN.h"
#include "cblas.h"

using namespace sgdtk;

DenseVectorN::DenseVectorN(const VectorN& source)
{
    this->from(source);
}

DenseVectorN::DenseVectorN(const DenseVectorN& dv)
{
    std::copy(dv.x.begin(), dv.x.end(), this->x.begin());
}
DenseVectorN::DenseVectorN()
{

}

DenseVectorN::DenseVectorN(int length)
{
    x.resize(length);
}

DenseVectorN::DenseVectorN(const std::vector<double>& x)
{
    std::copy(x.begin(), x.end(), this->x.begin());
}

DenseVectorN& DenseVectorN::operator=(const VectorN &v)
{
    if (&v != this)
    {
        from(v);
    }
    return *this;
}

DenseVectorN& DenseVectorN::operator=(const DenseVectorN &dv)
{
    if (&dv != this)
    {
        std::copy(dv.x.begin(), dv.x.end(), this->x.begin());
    }
    return *this;
}

DenseVectorN::~DenseVectorN() {}
void DenseVectorN::add(Offset offset)
{
    if (offset.first > x.size())
    {
        std::ostringstream oss;
        oss << "Index out of bounds! " << offset.first << ". Max is " << x.size();
        throw Exception(oss.str());
    }
    x[offset.first] = offset.second;
}

Offsets DenseVectorN::getNonZeroOffsets() const
{
    Offsets offsetList;
    for (int i = 0, sz = x.size(); i < sz; ++i)
    {
        if (x[i] != 0.0)
        {
            offsetList.push_back(Offset(i, x[i]));
        }
    }
    return offsetList;
}


void DenseVectorN::from(const VectorN &source)
{
    int length = source.length();
    x.resize(length, 0.0);
    Offsets offsets = source.getNonZeroOffsets();
    for (Offset offset : offsets)
    {
        x[offset.first] = offset.second;
    }
}

void DenseVectorN::organize()
{

}

#include <iostream>
double DenseVectorN::mag() const
{
    const double* v1 = &x[0];
    return (double)cblas_ddot(x.size(), v1, 1, v1, 1);
/*
    auto acc = 0.0;
    for (auto v : x)
    {
        acc += v * v;
    }
    return acc;
*/
}


void DenseVectorN::scale(double scalar)
{
    double* v1 = &x[0];
    cblas_dscal(x.size(), scalar, v1, 1);
    /*
     for (int i = 0, sz = x.size(); i < sz; ++i)
    {
        x[i] *= scalar;
    }
     */
}

double DenseVectorN::dot(const VectorN& vec) const
{
    if (vec.getType() == DENSE)
    {
        return ddot((const DenseVectorN&)vec);
    }
    auto acc = 0.;
    for (auto offset : vec.getNonZeroOffsets())
    {
        acc += x[offset.first] * offset.second;
    }
    return acc;
}

double DenseVectorN::ddot(const DenseVectorN& vec) const
{
    const auto dvec = (DenseVectorN) vec;
    const double *v1 = &x[0];
    const double *v2 = &(dvec.x[0]);
    return cblas_ddot(x.size(), v1, 1, v2, 1);
}
