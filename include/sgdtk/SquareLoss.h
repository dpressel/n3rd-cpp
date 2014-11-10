#ifndef __SGDTK_SQUARE_LOSS_H__
#define __SGDTK_SQUARE_LOSS_H__
#include "sgdtk/Loss.h"

namespace sgdtk
{


/**
 * Square loss
 *
 * @author dpressel
 */
class SquareLoss : public Loss
{
public:
    SquareLoss() {}
    ~SquareLoss() {}

    /**
     * Square loss
     * @param p predicted
     * @param y actual
     * @return loss
     */
    double loss(double p, double y) const
    {
        double z = p * y;
        double d = 1 - z;
        return 0.5 * d * d;
    }
    double dLoss(double p, double y) const
    {
        double z = p * y;
        double d = 1 - z;
        return -y * d;
    }
};
}

#endif