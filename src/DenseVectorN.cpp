#include "sgdtk/DenseVectorN.h"
#include "cblas.h"

using namespace sgdtk;

DenseVectorN::DenseVectorN(const VectorN& source)
{
    this->from(source);
}

DenseVectorN::DenseVectorN(const DenseVectorN& dv)
{
    x = dv.x;
    //std::copy(dv.x.begin(), dv.x.end(), this->x.begin());
}
DenseVectorN::DenseVectorN()
{

}

DenseVectorN::DenseVectorN(int length)
{
    x.resize({length});
}

DenseVectorN::DenseVectorN(const Tensor& x)
{
    this->x = x;
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
        this->x = dv.x;
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
    x.resize({length}, 0.0);
    Offsets offsets = source.getNonZeroOffsets();
    for (Offset offset : offsets)
    {
        x[offset.first] = offset.second;
    }
}

void DenseVectorN::organize()
{

}

double DenseVectorN::mag() const
{
#ifdef USE_BLAS
    const double* v1 = &x[0];
    return cblas_ddot(x.size(), v1, 1, v1, 1);
#else
    auto acc = 0.0;
    for (auto v : x)
    {
        acc += v * v;
    }
    return acc;
#endif
}

void DenseVectorN::scale(double scalar)
{
#ifdef USE_BLAS
    double* v1 = &x[0];
    cblas_dscal(x.size(), scalar, v1, 1);
#else
    for (int i = 0, sz = x.size(); i < sz; ++i)
    {
        x[i] *= scalar;
    }
#endif
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
#ifdef USE_BLAS
    const double *v1 = &x[0];
    const double *v2 = &(vec.x[0]);
    return cblas_ddot(x.size(), v1, 1, v2, 1);
#else
    auto acc = 0.;
    for (int i = 0, sz = x.size(); i < sz; ++i)
    {
        acc += vec.x[i] * x[i];
    }
    return acc;
#endif
}

extern "C" SGDTK_DVN sgdtk_DenseVectorN_create(int length)
{
    return new DenseVectorN(length);
}
extern "C" void sgdtk_DenseVectorN_destroy(SGDTK_DVN p)
{
    delete (DenseVectorN*)p;
}


extern "C" SGDTK_DVN sgdtk_DenseVectorN_copyOfDense(SGDTK_DVN other)
{
    return new DenseVectorN(*(DenseVectorN*)other);
}
extern "C" int sgdtk_DenseVectorN_length(SGDTK_DVN self)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    return dvn->length();
}
extern "C" void sgdtk_DenseVectorN_addOffset(SGDTK_DVN self, int offset, double value)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    dvn->add(Offset(offset, value));
}
extern "C" double sgdtk_DenseVectorN_mag(SGDTK_DVN self)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    return dvn->mag();
}
extern "C" void sgdtk_DenseVectorN_update(SGDTK_DVN self, int i, double v)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    dvn->update(i, v);
}
extern "C" void sgdtk_DenseVectorN_set(SGDTK_DVN self, int i , double v)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    dvn->set(i, v);
}
extern "C" void sgdtk_DenseVectorN_scale(SGDTK_DVN self, double scalar)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    dvn->scale(scalar);
}
extern "C" double sgdtk_DenseVectorN_at(SGDTK_DVN self, int i)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    return dvn->at(i);
}
//extern "C" double sgdtk_DenseVectorN_dot(SGDTK_DVN, void*);
extern "C" double sgdtk_DenseVectorN_ddot(SGDTK_DVN self, SGDTK_DVN other)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    DenseVectorN* dvn2 = (DenseVectorN*)other;
    return dvn->dot(*dvn2);
}
extern "C" void sgdtk_DenseVectorN_resetFromDense(SGDTK_DVN self, SGDTK_DVN other)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    DenseVectorN* dvn2 = (DenseVectorN*)other;
    *dvn = *dvn2;
}
//extern "C" void sgdtk_DenseVectorN_resetFrom(void* vn)
//{
//}
extern "C" void sgdtk_DenseVectorN_organize(SGDTK_DVN self)
{
    DenseVectorN* dvn = (DenseVectorN*)self;
    dvn->organize();
}
