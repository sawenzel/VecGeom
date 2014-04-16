/**
 * @file backend.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

#include "base/global.h"

#ifdef VECGEOM_NVCC
#include "backend/cuda/backend.h"
#else
#include "backend/vector/backend.h"
#endif

#endif // VECGEOM_BACKEND_BACKEND_H_