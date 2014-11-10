#ifndef __SGDTK_MODEL_H__
#define __SGDTK_MODEL_H__

#include "sgdtk/Types.h"
#include "sgdtk/FeatureVector.h"
namespace sgdtk
{


/**
 * Model for classification
 *
 * @author dpressel
 */
class Model
{
public:
    Model() {}
    virtual ~Model() {}

    /**
     * Load the model from the stream
     * @param file Model file to load from
     * @throws Exception
     */
    virtual void load(String file) = 0;

    /**
     * Save the model to a stream
     * @param file model to save to
     * @throws Exception
     */
    virtual void save(String file) = 0;

    /**
     * Predict y given feature vector
     * @param fv feature vector
     * @return prediction
     */
    virtual double predict(const FeatureVector* fv) const = 0;

    /**
     * Create a deep copy of this
     * @return clone
     */
    virtual Model* prototype() const = 0;
};
}
#endif