#ifndef __SGDTK_LEARNER_H__
#define __SGDTK_LEARNER_H__

#include "sgdtk/Model.h"
#include "sgdtk/Metrics.h"

namespace sgdtk
{

/**
 * A trainer for unstructured classification.
 *
 * @author dpressel
 */
class Learner
{
public:
    Learner() {}
    virtual ~Learner() {}

    /**
     * Create an empty but initialized model, with the length of the feature vector given
     * @param wlength The length of the feature vector
     * @return An empty but initialized model
     */
    virtual Model* create(int wlength) = 0;

    virtual void preprocess(Model* model, const std::vector<FeatureVector*>& sample) = 0;
    /**
     * Train on a single pass
     * @param model The model to update
     * @param trainingExamples The training examples
     * @return The updated model
     */
    virtual Model* trainEpoch(Model* model,
        const std::vector<FeatureVector*>& trainingExamples) = 0;

    virtual void trainOne(Model* model, const FeatureVector* fv) = 0;

    /**
     * Evaluate a set of examples
     * @param model The model to use for evaluation
     * @param testingExamples The examples
     * @param metrics Metrics to add to
     */
    virtual void eval(Model* model,
        const std::vector<FeatureVector*>& testingExamples,
        Metrics& metrics) = 0;

    /**
     * Evaluate a single instance
     * @param model The model to use for evaluation
     * @param fv The feature vector
     * @param metrics Metrics to add to
     * @return labels or probabilities from learner for this example
     */
    virtual double evalOne(Model* model,
        const FeatureVector* fv,
        Metrics& metrics) = 0;


    
};

}

#endif