#ifndef __N3RD_CLASSNLLLOSS_H__
#define __N3RD_CLASSNLLLOSS_H__

#include "sgdtk/Loss.h"
#include <cmath>

namespace n3rd
{


/**
 * Log loss function
 * @author dpressel
 */
    class ClassNLLLoss : public sgdtk::Loss
    {
    public:
        ClassNLLLoss()
        { }

        ~ClassNLLLoss()
        { }

        double loss(double p, double y) const
        {
            return -p;
        }

        double dLoss(double p, double y) const
        {
            return -1;
        }

    };
}

#endif