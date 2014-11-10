#ifndef __SGDTK_HINGE_LOSS_H__
#define __SGDTK_HINGE_LOSS_H__

#include "sgdtk/Loss.h"
#include <limits>

namespace sgdtk
{


/**
 * Hinge loss function
 *
 * @author dpressel
 */
class HingeLoss : public Loss
{
public:
	
	HingeLoss() {}
	~HingeLoss() {}
	
	/**
     * Hinge loss function
     * @param p prediction
     * @param y actual
     * @return loss
     */
	double loss(double p, double y) const
	{
		return std::max<double>(0., 1 - p * y);
	}

	/**
     * Derivative of loss function
     * @param p prediction
     * @param y actual
     * @return derivative
     */
	double dLoss(double p, double y) const
	{
		if (loss(p, y) == 0.)
			return 0.;
		return -y;
	}
};

}
#endif