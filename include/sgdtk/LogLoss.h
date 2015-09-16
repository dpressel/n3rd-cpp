#ifndef __SGDTK_LOG_LOSS_H__
#define __SGDTK_LOG_LOSS_H__

#include "sgdtk/Loss.h"
#include <cmath>

namespace sgdtk
{


/**
 * Log loss function
 * @author dpressel
 */
    class LogLoss : public Loss
    {
    public:
        LogLoss()
        { }

        ~LogLoss()
        { }

        /**
         * Math.log(1 + Math.exp(-z))
         * @param p prediction
         * @param y actual
         * @return loss
         */
        double loss(double p, double y) const
        {
            auto z = p * y;
            return z > 18 ? std::exp(-z) : z < -18 ? -z : std::log(1 + std::exp(-z));
        }

        /**
         * -y / (1 + Math.exp(z))
         * @param p prediction
         * @param y actual
         * @return derivative
         */
        double dLoss(double p, double y) const
        {
            auto z = p * y;
            return z > 18 ? -y * std::exp(-z) : z < -18 ? -y : -y / (1 + std::exp(z));
        }

    };
}

#endif