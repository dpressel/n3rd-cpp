#ifndef __SGDTK_CPP_ROBBINSMONROUPDATESCHEDULE_H__
#define __SGDTK_CPP_ROBBINSMONROUPDATESCHEDULE_H__

#include "sgdtk/LearningRateSchedule.h"

namespace sgdtk
{
class RobbinsMonroUpdateSchedule : public LearningRateSchedule
{
    long numSeenTotal;
    double eta0;
    double lambda;

public:
    RobbinsMonroUpdateSchedule()
    {

    }
    ~RobbinsMonroUpdateSchedule()
    {

    }
    void reset(double eta0, double lambda)
    {
        this->lambda = lambda;
        this->eta0 = eta0;
        this->numSeenTotal = 0;
    }

    double update()
    {
        auto eta = eta0 / (1 + lambda * eta0 * numSeenTotal);
        ++numSeenTotal;
        return eta;
    }

};
}

#endif
