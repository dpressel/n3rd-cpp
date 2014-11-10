#ifndef __SGDTK_LOSS_H__
#define __SGDTK_LOSS_H__

namespace sgdtk
{
class Loss
{
public:
	Loss() {}
	virtual ~Loss() {}
	virtual double loss(double p, double y) const = 0;
	virtual double dLoss(double p, double y) const = 0;
};
}
#endif