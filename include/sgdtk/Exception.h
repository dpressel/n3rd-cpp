#ifndef __SGDTK_EXCEPTION_H__
#define __SGDTK_EXCEPTION_H__


#include "sgdtk/Types.h"
namespace sgdtk
{
    /**
     * Absolutely simple base exception
     *
     * @author dpressel
     */
	class Exception
	{
		String message;
	public:

		Exception()
		{

		}
		Exception(String message);

		~Exception()
		{

		}
		String getMessage() const;
	};
}

#endif