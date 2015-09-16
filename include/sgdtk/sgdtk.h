#ifndef __SGDTK_H__
#define __SGDTK_H__

/**
 * Include this file in your application to avoid having to do piecemeal includes
 */
#include "sgdtk/Exception.h"
#include "sgdtk/FeatureProvider.h"
#include "sgdtk/FeatureVector.h"
#include "sgdtk/HingeLoss.h"
#include "sgdtk/Learner.h"
#include "sgdtk/LinearModel.h"
#include "sgdtk/LogLoss.h"
#include "sgdtk/Loss.h"
#include "sgdtk/Metrics.h"
#include "sgdtk/Model.h"
#include "sgdtk/Params.h"
#include "sgdtk/SGDLearner.h"
#include "sgdtk/SquaredLoss.h"
#include "sgdtk/SquaredHingeLoss.h"
#include "sgdtk/SVMLightFileFeatureProvider.h"
#include "sgdtk/Types.h"
#include "sgdtk/VectorN.h"
#include "sgdtk/DenseVectorN.h"
#include "sgdtk/SparseVectorN.h"
#include "sgdtk/ModelFactory.h"
#include "sgdtk/LinearModelFactory.h"
#include "sgdtk/LearningRateSchedule.h"
#include "sgdtk/RobbinsMonroUpdateSchedule.h"
#include "sgdtk/FixedLearningRate.h"
#include "sgdtk/ConsumerProducerQueue.h"

#endif