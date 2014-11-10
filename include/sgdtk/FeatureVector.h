#ifndef __SGDTK_FEATURE_VECTOR_H__
#define __SGDTK_FEATURE_VECTOR_H__

#include "sgdtk/Types.h"
namespace sgdtk
{
class FeatureVector
{
    Offsets rep;
    size_t size;
    Number label;
    
public:

    /**
     * Constructor for feature vectors that are ground truth
     * @param y label
     * @param size feature vector width
     */
    FeatureVector(size_t sz, Number labelValue = 0.) :
        size(sz), label(labelValue) {}

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
        return size;
    }

};

}
#endif