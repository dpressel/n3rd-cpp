//
// Created by Daniel on 9/15/2015.
//

#ifndef __SGDTK_CPP_FIXEDLEARNINGRATE_H__
#define __SGDTK_CPP_FIXEDLEARNINGRATE_H__

#include "sgdtk/LearningRateSchedule.h"
namespace sgdtk
{
    class FixedLearningRate : public LearningRateSchedule
    {
    public:
        FixedLearningRate() {}
        ~FixedLearningRate() {}
        double eta;

        void reset(double eta0, double lambda)
        {
            this->eta = eta0;
        }
        double update()
        {
            return eta;
        }
    };
}

#endif
