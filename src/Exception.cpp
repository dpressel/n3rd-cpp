#include "sgdtk/Exception.h"

using namespace sgdtk;

Exception::Exception(String s) : message(s)
{
}

String Exception::getMessage() const
{
    return message;
}