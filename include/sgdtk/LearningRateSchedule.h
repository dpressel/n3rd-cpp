#ifndef __SGDTK_CPP_LEARNING_RATE_SCHEDULE_H__
#define __SGDTK_CPP_LEARNING_RATE_SCHEDULE_H__

namespace sgdtk
{
    class LearningRateSchedule
    {
    public:
        LearningRateSchedule() {}
        virtual ~LearningRateSchedule() {}

        virtual void reset(double eta0, double lambda) = 0;
        virtual double update() = 0;
    };
}

#endif