#include "sgdtk/DenseVectorN.h"

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
    for (int i = 0, sz = offsets.size(); i < sz; ++i)
    {
        x[offsets[i].first] = offsets[i].second;
    }
}

void DenseVectorN::organize()
{

}

double DenseVectorN::mag() const
{
    double acc = 0.0;
    for (int i = 0, sz = x.size(); i < sz; ++i)
    {
        acc += x[i] * x[i];
    }
    return acc;
}

double DenseVectorN::dot(const VectorN& vec)
{
    double acc = 0.;
    for (int i = 0, sz = x.size(); i < sz; ++i)
    {
        acc += x[i] * vec.at(i);
    }
    return acc;
}