/**
 * @file vector/implementation.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_VECTOR_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_VECTOR_IMPLEMENTATION_H_

#include "base/global.h"

#if defined(VECGEOM_VC)
#include "backend/vc/implementation.h"
#elif defined(VECGEOM_CILK)
#include "backend/cilk/implementation.h"
#else
#include "backend/scalar/implementation.h"
#endif

#endif // VECGEOM_BACKEND_VECTOR_IMPLEMENTATION_H_