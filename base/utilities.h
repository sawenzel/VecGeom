#ifndef VECGEOM_BASE_UTILITIES_H_
#define VECGEOM_BASE_UTILITIES_H_

#ifndef __CUDACC__
#define VECGEOM_STD_CXX11
#define VECGEOM_CUDA_HEADER_DEVICE
#define VECGEOM_CUDA_HEADER_HOST
#define VECGEOM_CUDA_HEADER_BOTH
#else // __CUDACC__
#define VECGEOM_NVCC
#define VECGEOM_CUDA_HEADER_DEVICE __device__
#define VECGEOM_CUDA_HEADER_HOST __host__
#define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
#endif // __CUDACC__

#ifdef __INTEL_COMPILER
#define VECGEOM_INLINE inline
#else // __INTEL_COMPILER
#ifdef __GNUC__
#define VECGEOM_INLINE inline __attribute__((always_inline))
#else // __GNUC__
#define VECGEOM_INLINE inline
#endif // __GNUC__
#endif // __INTEL_COMPILER

#ifdef VECGEOM_STD_CXX11
constexpr int kAlignmentBoundary = 32;
#else // VECGEOM_STD_CXX11
const int kAlignmentBoundary = 32;
#endif // VECGEOM_STD_CXX11

#endif // VECGEOM_BASE_UTILITIES_H_