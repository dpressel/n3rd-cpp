#ifndef __SGDTK_SQUARED_LOSS_H__
#define __SGDTK_SQUARED_LOSS_H__

#include "sgdtk/Loss.h"

namespace sgdtk
{


/**
 * Square loss
 *
 * @author dpressel
 */
    class SquaredLoss : public Loss
    {
    public:
        SquaredLoss()
        { }

        ~SquaredLoss()
        { }

        /**
         * Square loss
         * @param p predicted
         * @param y actual
         * @return loss
         */
        double loss(double p, double y) const
        {
            auto d = p - y;
            return 0.5 * d * d;
        }

        double dLoss(double p, double y) const
        {
            return (p - y);
        }
    };
}

#endif