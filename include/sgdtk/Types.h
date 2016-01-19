#ifndef __SGDTK_TYPES_H__
#define __SGDTK_TYPES_H__

#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <sys/time.h>
#include <algorithm>
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
        if (gettimeofday(&time, NULL))
        {
            return -1000000;
        }
        return (double) time.tv_sec + (double) time.tv_usec * .000001;
    }

    inline StringArray split(const String &s, char delim = ' ')
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

    static inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    static inline std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
    }
    template<typename T>
    T valueOf(String s)
    {
        std::istringstream iss(s);
        T t;
        iss >> t;
        return t;

    }

    template<typename T>
    String toString(const T &t)
    {
        std::ostringstream oss;
        oss << t;
        return oss.str();
    }

    // These 3 methods are borrowed from CODA: https://github.com/mdaus/coda-oss
    inline bool isBigEndianSystem()
    {
        // This is an endian test
        int intVal = 1;
        unsigned char* endianTest = (unsigned char*) & intVal;
        return endianTest[0] != 1;
    }

    inline void byteSwap(void* buffer,
                         unsigned short elemSize,
                         size_t numElems)
    {
        unsigned char* bufferPtr = static_cast<unsigned char*>(buffer);
        if (!bufferPtr || elemSize < 2 || !numElems)
            return;

        unsigned short half = elemSize >> 1;
        size_t offset = 0, innerOff = 0, innerSwap = 0;

        for(size_t i = 0; i < numElems; ++i, offset += elemSize)
        {
            for(unsigned short j = 0; j < half; ++j)
            {
                innerOff = offset + j;
                innerSwap = offset + elemSize - 1 - j;

                std::swap(bufferPtr[innerOff], bufferPtr[innerSwap]);
            }
        }
    }

    template <typename T> T byteSwap(T val)
    {
        size_t size = sizeof(T);
        T out;

        unsigned char* cOut = reinterpret_cast<unsigned char*>(&out);
        unsigned char* cIn = reinterpret_cast<unsigned char*>(&val);
        for (int i = 0, j = size - 1; i < j; ++i, --j)
        {
            cOut[i] = cIn[j];
            cOut[j] = cIn[i];
        }
        return out;
    }
}

#endif