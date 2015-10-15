#ifndef __SGDTK_CPP_DENSEVECTORN_H__
#define __SGDTK_CPP_DENSEVECTORN_H__

#include "sgdtk/Exception.h"
#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"
#include "Params.h"
#include "sgdtk/Tensor.h"

namespace sgdtk
{
    class DenseVectorN : public VectorN
    {

    public:
        Tensor x;
        DenseVectorN(const VectorN& source);
        DenseVectorN(const DenseVectorN& dv);
        DenseVectorN();

        DenseVectorN(int length);

        DenseVectorN(const Tensor& x);

        DenseVectorN& operator=(const VectorN &v);

        DenseVectorN& operator=(const DenseVectorN &dv);

        ~DenseVectorN();

        void resize(int length)
        {
            x.resize({length});
        }

        int length() const
        {
            return x.size();
        }

        void add(Offset offset);

        double mag() const;

        double update(int i, double v)
        {
            x[i] += v;
            return x[i];
        }

        void set(int i, double v)
        {
            x[i] = v;
        }

        void scale(double scalar);

        Offsets getNonZeroOffsets() const;

        double at(int i) const
        {
            return x[i];
        }

        Tensor& getX()
        {
            return x;
        }
        const Tensor& getX() const
        {
            return x;
        }

        double dot(const VectorN& vec) const;

        double ddot(const DenseVectorN& vec) const;

        void from(const VectorN& source);

        void organize();

        const Type getType() const
        {
            return DENSE;
        }
    };
}

extern "C"
{
    typedef void* SGDTK_DVN;

    SGDTK_DVN sgdtk_DenseVectorN_create(int);
    void  sgdtk_DenseVectorN_destroy(SGDTK_DVN);
    //void  sgdtk_DenseVectorN_copyOf(SGDTK_DVN, void*);
    SGDTK_DVN sgdtk_DenseVectorN_copyOfDense(SGDTK_DVN other);
    int sgdtk_DenseVectorN_length(SGDTK_DVN);
    void sgdtk_DenseVectorN_addOffset(SGDTK_DVN,int, double);
    double sgdtk_DenseVectorN_mag(SGDTK_DVN);
    void sgdtk_DenseVectorN_update(SGDTK_DVN, int, double);
    void sgdtk_DenseVectorN_set(SGDTK_DVN, int, double);
    void sgdtk_DenseVectorN_scale(SGDTK_DVN, double);
    double sgdtk_DenseVectorN_at(SGDTK_DVN, int);
    //double sgdtk_DenseVectorN_dot(SGDTK_DVN, void*);
    double sgdtk_DenseVectorN_ddot(SGDTK_DVN, SGDTK_DVN);
    void sgdtk_DenseVectorN_resetFromDense(SGDTK_DVN, SGDTK_DVN);
    //void sgdtk_Dense_resetFrom(void*);
    void sgdtk_DenseVectorN_organize(SGDTK_DVN);

}


#endif
