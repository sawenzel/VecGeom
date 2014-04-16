/**
 * @file implementation.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_IMPLEMENTATION_H_
#define VECGEOM_BACKEND_IMPLEMENTATION_H_

#include "base/global.h"

#ifdef VECGEOM_NVCC
#include "backend/cuda/implementation.h"
#else
#include "backend/vector/implementation.h"
#endif

#endif // VECGEOM_BACKEND_IMPLEMENTATION_H_