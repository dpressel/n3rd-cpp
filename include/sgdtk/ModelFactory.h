#ifndef __SGDTK_CPP_MODELFACTORY_H__
#define __SGDTK_CPP_MODELFACTORY_H__

#include "sgdtk/Model.h"
#include "sgdtk/Exception.h"

namespace sgdtk
{
    class ModelFactory
    {
    public:
        ModelFactory() {}
        virtual ~ModelFactory() {}

        virtual Model* newInstance(void* p) const throw(Exception) = 0;

    };
}

#endif
