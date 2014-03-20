/**
 * @file backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

#include "base/global.h"

#if (defined(VECGEOM_VC) && !defined(VECGEOM_NVCC))
#include "backend/vc/backend.h"
#endif

#if (defined(VECGEOM_CILK) && !defined(VECGEOM_NVCC))
#include "backend/cilk/backend.h"
#endif

#ifdef VECGEOM_NVCC
#include "backend/cuda/backend.h"
#endif

#endif // VECGEOM_BACKEND_BACKEND_H_