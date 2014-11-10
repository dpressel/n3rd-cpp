#ifndef __SGDTK_FEATURE_PROVIDER_H__
#define __SGDTK_FEATURE_PROVIDER_H__

#include "sgdtk/FeatureVector.h"

namespace sgdtk
{
/**
 * Streaming interface for getting FeatureVector
 *
 * @author dpressel
 */
class FeatureProvider
{
public:
	FeatureProvider() {}
	virtual ~FeatureProvider() {}

	virtual int getLargestVectorSeen() = 0;
    /**
     * Get the next feature vector from the source
     * @return feature vector or null of end of stream reached
     * @throws IOException
     */
	virtual FeatureVector* next() = 0;
};
}
#endif