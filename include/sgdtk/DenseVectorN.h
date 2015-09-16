#ifndef __SGDTK_CPP_DENSEVECTORN_H__
#define __SGDTK_CPP_DENSEVECTORN_H__

#include "sgdtk/Exception.h"
#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"
#include "Params.h"

namespace sgdtk
{
    class DenseVectorN : public VectorN
    {

    public:
        std::vector<double> x;
        DenseVectorN(const VectorN& source);
        DenseVectorN(const DenseVectorN& dv);
        DenseVectorN();

        DenseVectorN(int length);

        DenseVectorN(const std::vector<double>& x);

        DenseVectorN& operator=(const VectorN &v);

        DenseVectorN& operator=(const DenseVectorN &dv);

        ~DenseVectorN();

        void resize(int length)
        {
            x.resize(length);
        }
/*
        const double operator[](int at) const
        {
            return x[at];
        }
        double& operator[](int at)
        {
            return x[at];
        }
*/
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

        void scale(double scalar)
        {
            for (int i = 0, sz = x.size(); i < sz; ++i)
            {
                x[i] *= scalar;
            }
        }

        Offsets getNonZeroOffsets() const;

        double at(int i) const
        {
            return x[i];
        }

        double dot(const VectorN& vec);


        void from(const VectorN& source);

        void organize();
    };
}
#endif
