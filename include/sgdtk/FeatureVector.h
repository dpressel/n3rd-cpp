#ifndef __SGDTK_FEATURE_VECTOR_H__
#define __SGDTK_FEATURE_VECTOR_H__

#include "sgdtk/Types.h"
namespace sgdtk
{
class FeatureVector
{
    Offsets rep;
    Number label;
    
public:

    /**
     * Constructor for feature vectors that are ground truth
     * @param y label
     */
    FeatureVector(Number labelValue = 0.) :
        label(labelValue) {}

    ~FeatureVector() {}

    /**
     * Add a new offset to the feature vector (must not exceed size)
     * @param offset
     */
    void add(Offset offset)
    {
        rep.push_back(offset);
    }

    /**
     * Get the label
     * @return return label
     */
    Number getY() const
    {
        return label;
    }
    void setY(double label)
    {
        this->label = label;
    }

    /**
     * Get all non-zero values and their indices
     * @return
     */
    const Offsets& getNonZeroOffsets() const
    {
        return rep;
    }

    /**
     * Length of feature vector
     * @return
     */
    size_t length() const
    {
        int sz = rep.size();
        return sz == 0 ? 0: (rep[sz - 1].first + 1);
    }

};

}
#endif