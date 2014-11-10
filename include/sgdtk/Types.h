#ifndef __SGDTK_TYPES_H__
#define __SGDTK_TYPES_H__

#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <sys/time.h>

/**
 * Provide some useful aliases and definitions to make the C++ more similar
 * to the Java code.
 *
 */
namespace sgdtk
{
    typedef std::vector<std::string> StringArray;
    typedef std::string String;
    typedef double Number;
    typedef std::pair<int, Number> Offset;
    typedef std::vector<Offset> Offsets;
    typedef std::vector<Number> DenseVector;


    inline double currentTimeSeconds()
    {
        struct timeval time;
        if (gettimeofday(&time,NULL))
        {
            return -1000000;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
    }

    inline StringArray split(const String& s, char delim = ' ')
    {
        StringArray elems;
        std::stringstream ss(s);
        String item;
        while (std::getline(ss, item, delim))
        {
            
            elems.push_back(item);
        }
        return elems;
    }

    template <typename T> T valueOf(String s)
    {
        std::istringstream iss(s);
        T t;
        iss >> t;
        return t;

    }
    template <typename T> String toString(const T& t)
    {
        std::ostringstream oss;
        oss << t;
        return oss.str();
    }
}

#endif