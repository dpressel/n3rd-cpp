#ifndef __SGDTK_FEATURE_VECTOR_H__
#define __SGDTK_FEATURE_VECTOR_H__

#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"
#include "sgdtk/SparseVectorN.h"

namespace sgdtk
{
    class FeatureVector
    {
        VectorN *x;
        double y;

    public:

        const double UNLABELED = 0x0.0000000000001P-1022;

        /**
         * Constructor for feature vectors that are ground truth
         * @param y label
         */
        FeatureVector(Number labelValue, VectorN *vectorN = NULL) :
                y(labelValue)
        {

            x = (vectorN == NULL) ? (new SparseVectorN()): (vectorN);
        }


        ~FeatureVector()
        {
            delete x;
        }

        /**
         * Add a new offset to the feature vector (must not exceed size)
         * @param offset
         */
        void add(Offset offset)
        {
            x->add(offset);
        }

        /**
         * Get the label
         * @return return label
         */
        Number getY() const
        {
            return y;
        }

        const VectorN* getX() const
        {
            return x;
        }

        VectorN* getX()
        {
            return x;
        }

        void setY(double label)
        {
            this->y = label;
        }

        /**
         * Get all non-zero values and their indices
         * @return
         */
        Offsets getNonZeroOffsets() const
        {
            return x->getNonZeroOffsets();
        }

        /**
         * Length of feature vector
         * @return
         */
        size_t length() const
        {
            return x->length();
        }

    };

}
#endif