#include "sgdtk/Params.h"

using namespace sgdtk;

Params::Params(int argc, char **argv)
{

    for (int i = 1; i < argc; ++i)
    {
        String s = argv[i];
        if (s[0] == '-')
        {
            String option = s.substr(s.find_first_not_of('-'));
            String value = argv[++i];
            args[option] = value;

        }
    }
}

String Params::operator()(String arg, String def) const
{
    ArgMap::const_iterator it = this->args.find(arg);
    if (it == args.end())
        return def;

    return it->second;
}


