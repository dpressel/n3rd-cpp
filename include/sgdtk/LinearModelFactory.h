#ifndef __SGDTK_CPP_LINEARMODELFACTORY_H__
#define __SGDTK_CPP_LINEARMODELFACTORY_H__

#include "sgdtk/Model.h"
#include "sgdtk/Exception.h"
#include "sgdtk/LinearModel.h"
#include "sgdtk/AdagradLinearModel.h"
namespace sgdtk
{
    class LinearModelFactory : public ModelFactory
    {
        String type;
    public:

        LinearModelFactory(String type = "sgd")
        {
            this->type = type;
        }
        virtual ~LinearModelFactory() {}

        virtual Model* newInstance(void* p) const throw(Exception)
        {
            auto wlength = (long)p;
            return (type == "adagrad") ? (new AdagradLinearModel(wlength)): (new LinearModel(wlength));
        }

    };
}

#endif
