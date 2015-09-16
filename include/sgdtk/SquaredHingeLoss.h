#ifndef __SGDTK_SQUARE_LOSS_H__
#define __SGDTK_SQUARE_LOSS_H__

#include "sgdtk/Loss.h"

namespace sgdtk
{


/**
 * Squared Hinge loss
 *
 * @author dpressel
 */
    class SquaredHingeLoss : public Loss
    {
    public:
        SquaredHingeLoss()
        { }

        ~SquaredHingeLoss()
        { }

        /**
         * Square loss
         * @param p predicted
         * @param y actual
         * @return loss
         */
        double loss(double p, double y) const
        {
            double z = p * y;
            if (z > 1.0)
                return 0.0;
            double d = 1 - z;
            return 0.5 * d * d;
        }

        double dLoss(double p, double y) const
        {
            double z = p * y;
            if (z > 1.0)
                return 0.0;
            double d = 1 - z;
            return -y * d;
        }
    };
}

#endif