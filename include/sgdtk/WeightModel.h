#ifndef __SGDTK_WEIGHT_MODEL_H__
#define __SGDTK_WEIGHT_MODEL_H__

#include "sgdtk/Model.h"
#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"
#include "sgdtk/DenseVectorN.h"

namespace sgdtk
{

/**
 * Linear model for classification
 *
 * @author dpressel
 */
    class WeightModel : public Model
    {

    public:

        WeightModel() {}
        virtual ~WeightModel() {}

        virtual double mag() const = 0;

        virtual void updateWeights(const VectorN* vectorN, double eta, double lambda, double dLoss, double y) = 0;



    };
}

#endif