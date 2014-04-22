//===-- package/ClassName.h - Instruction class definition -------*- C++ -*-===//
//
// VecGeom Project - all rights reserved.
//
// This file is distributed under the LGPL License. See LICENSE.TXT for details.
//
// 140411 Guilherme Lima - Created
//
//===----------------------------------------------------------------------===//
///
/// \file  parameters.h
/// \brief This file contains the declaration of the Parameters class, which represents
///        a generic set of parameters.  Originally developed to hold shape parameters,
///        but it can be reused in many other contexts.
/// 
//===----------------------------------------------------------------------===//

#ifndef VECGEOM_BASE_PARAMETERS_H_
#define VECGEOM_BASE_PARAMETERS_H_

#include "base/global.h"
#include <cassert>

namespace VECGEOM_NAMESPACE {

/// @
/// \brief  A generic class to hold typical parameters of type T
///
template <const unsigned size, typename T=Precision>
class Parameters {

private:
	// parameters
	T _params[size];

public:
	/// Accessors
    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	T par(const unsigned index) const {
		assert(index<size);
		return _params[index];
	}

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	void set(const unsigned index, T const& value) {
		assert(index<size);
		_params[index] = value;
	}

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	T const & operator[](const unsigned index) const {
		assert(index<size);
		return _params[index];
	}

    VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
	T& operator[](const unsigned index) {
		assert(index<size);
		return _params[index];
	}
};

} // End of namespace
#endif
