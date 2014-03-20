/**
 * @file implementation.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_IMPLEMENTATION_H_

#include "base/global.h"

#if (defined(VECGEOM_VC) && !defined(VECGEOM_NVCC))
#include "backend/vc/implementation.h"
#endif

#if (defined(VECGEOM_CILK) && !defined(VECGEOM_NVCC))
#include "backend/cilk/implementation.h"
#endif

#ifdef VECGEOM_NVCC
#include "backend/cuda/implementation.h"
#endif

#endif // VECGEOM_BACKEND_IMPLEMENTATION_H_