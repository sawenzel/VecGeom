/**
 * @file vector/backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_VECTOR_BACKEND_H_
#define VECGEOM_BACKEND_VECTOR_BACKEND_H_

#include "base/global.h"

#if defined(VECGEOM_VC)
#include "backend/vc/backend.h"
#elif defined(VECGEOM_CILK)
#include "backend/cilk/backend.h"
#else
#include "backend/scalar/backend.h"
#endif

#endif // VECGEOM_BACKEND_VECTOR_BACKEND_H_