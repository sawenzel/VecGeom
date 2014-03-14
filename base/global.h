/**
 * @file global.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_UTILITIES_H_
#define VECGEOM_BASE_UTILITIES_H_

#include <cmath>
#include "base/types.h"

#if (defined(__CUDACC__) || defined(__NVCC__))
  #define VECGEOM_NVCC
  #define VECGEOM_CUDA_HEADER_DEVICE __device__
  #define VECGEOM_CUDA_HEADER_HOST __host__
  #define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
  #define VECGEOM_CUDA_HEADER_GLOBAL __global__
#else // Not compiling for CUDA
  #define VECGEOM_CUDA_HEADER_DEVICE
  #define VECGEOM_CUDA_HEADER_HOST
  #define VECGEOM_CUDA_HEADER_BOTH
  #define VECGEOM_CUDA_HEADER_GLOBAL
#endif

#ifndef VECGEOM_CUDA // Set by compiler
  #define VECGEOM_STD_CXX11
#endif

#ifdef __INTEL_COMPILER
  #define VECGEOM_INTEL
  #define VECGEOM_INLINE inline
#else
  #include <mm_malloc.h>
  #if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__))
    #define VECGEOM_INLINE inline __attribute__((always_inline))
  #else // Clang (most likely)
    #define VECGEOM_INLINE inline
  #endif
#endif

#ifndef NULL
  #define NULL 0
#endif

namespace vecgeom {

const int kAlignmentBoundary = 32;
const double kDegToRad = M_PI/180.;
const double kRadToDeg = 180./M_PI;
const double kInfinity = INFINITY;
const double kTiny = 1e-20;
const double kTolerance = 1e-12;

} // End namespace vecgeom

#endif // VECGEOM_BASE_UTILITIES_H_