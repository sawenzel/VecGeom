/// \file vector/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

#include "base/Global.h"

#ifdef VECGEOM_NVCC
#include "backend/cuda/Backend.h"
#else
#include "backend/vector/Backend.h"
#endif

#endif // VECGEOM_BACKEND_BACKEND_H_