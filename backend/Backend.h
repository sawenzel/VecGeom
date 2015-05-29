/// \file vector/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_BACKEND_H_
#define VECGEOM_BACKEND_BACKEND_H_

#include "base/Global.h"

#ifdef VECGEOM_NVCC
#include "backend/cuda/Backend.h"
#elif defined(VECGEOM_VC)
#include "backend/vc/Backend.h"
//#include "backend/vcfloat/Backend.h"
#elif defined(VECGEOM_CILK)
#include "backend/cilk/Backend.h"
#elif defined(MIC_SIDE)
#include "backend/mic/Backend.h"
#else
#include "backend/scalar/Backend.h"
#endif

#endif // VECGEOM_BACKEND_BACKEND_H_
