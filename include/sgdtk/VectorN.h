#ifndef __SGDTK_VECTORN_H__
#define __SGDTK_VECTORN_H__

#include "sgdtk/Types.h"

namespace sgdtk
{
    class VectorN
    {
    public:
        enum Type { SPARSE, DENSE };
        virtual ~VectorN()
        { }

        virtual int length() const = 0;

        virtual void add(Offset offset) = 0;

        virtual void set(int i, double v) = 0;

        virtual double dot(const VectorN &vec) const = 0;

        virtual double sparseDot(const Offsets &offsets) const = 0;

        virtual Offsets getNonZeroOffsets() const = 0;

        virtual double at(int i) const = 0;

        virtual void from(const VectorN &source) = 0;

        virtual void organize() = 0;

        virtual const Type getType() const = 0;
    };
}
#endif