#ifndef __SGDTK_PARAMS_H__
#define __SGDTK_PARAMS_H__

#include "sgdtk/Types.h"

namespace sgdtk
{


    typedef std::map<String, String> ArgMap;

    class Params
    {

        ArgMap args;

    public:
        Params(int argc, char **argv);

        String operator()(String arg, String def = "") const;

    };

}
#endif