#ifndef VECGEOM_BASE_UTILITIES_H_
#define VECGEOM_BASE_UTILITIES_H_

#include <cmath>

namespace vecgeom {

#if (defined(__CUDACC__) || defined(__NVCC__))
  #define VECGEOM_NVCC
  #define VECGEOM_CUDA_HEADER_DEVICE __device__
  #define VECGEOM_CUDA_HEADER_HOST __host__
  #define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
#else // __CUDACC__ || __NVCC__
  #define VECGEOM_CUDA_HEADER_DEVICE
  #define VECGEOM_CUDA_HEADER_HOST
  #define VECGEOM_CUDA_HEADER_BOTH
#endif // __CUDACC__ || __NVCC__

#ifndef VECGEOM_CUDA // Set by compiler
  #define VECGEOM_STD_CXX11
#endif // VECGEOM_CUDA

#ifdef __INTEL_COMPILER
  #define VECGEOM_INLINE inline
#else // __INTEL_COMPILER
  #if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__))
    #define VECGEOM_INLINE inline __attribute__((always_inline))
  #else // __GNUC__
    #define VECGEOM_INLINE inline
  #endif // __GNUC__
#endif // __INTEL_COMPILER

const int kAlignmentBoundary = 32;
const double kDegToRad = M_PI/180.;
const double kRadToDeg = 180./M_PI;
const double kInfinity = INFINITY;
const double kTiny = 1e-20;
const double kGTolerance = 1e-9;

} // End namespace vecgeom

#endif // VECGEOM_BASE_UTILITIES_H_