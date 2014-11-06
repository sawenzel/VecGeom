/// \file Global.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_GLOBAL_H_
#define VECGEOM_BASE_GLOBAL_H_

#include <cassert>
#include <cmath>
#include <float.h>
#include <limits>
#include <stdio.h>

#define VECGEOM

#if __cplusplus >= 201103L
  #define VECGEOM_STD_CXX11
#endif

#if (defined(__CUDACC__) || defined(__NVCC__))
  // Compiling with nvcc
  #define VECGEOM_NVCC
  #ifdef __CUDA_ARCH__
    // Compiling device code
    #define VECGEOM_NVCC_DEVICE
  #endif
  #define VECGEOM_NAMESPACE vecgeom_cuda
  #define VECGEOM_CUDA_HEADER_HOST __host__
  #define VECGEOM_CUDA_HEADER_DEVICE __device__
  #define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
  #define VECGEOM_CUDA_HEADER_GLOBAL __global__
  #undef VECGEOM_VC
  #undef VECGEOM_VC_ACCELERATION
  #undef VECGEOM_CILK
  #undef VECGEOM_ROOT
  #undef VECGEOM_USOLIDS
  #undef VECGEOM_GEANT4
  #undef VECGEOM_BENCHMARK
#else
  // Not compiling with NVCC
  #define VECGEOM_NAMESPACE vecgeom
  #define VECGEOM_CUDA_HEADER_HOST
  #define VECGEOM_CUDA_HEADER_DEVICE
  #define VECGEOM_CUDA_HEADER_BOTH
  #define VECGEOM_CUDA_HEADER_GLOBAL
  #ifdef VECGEOM_CUDA
    // CUDA is enabled, but currently compiling regular C++ code.
    // This enables methods that interface between C++ and CUDA environments
    #define VECGEOM_CUDA_INTERFACE
  #endif
#endif

#ifdef __INTEL_COMPILER
  // Compiling with icc
  #define VECGEOM_INTEL
  #define VECGEOM_INLINE inline
#else
  // Functionality of <mm_malloc.h> is automatically included in icc
  #include <mm_malloc.h>
  #if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__clang__) && !defined(__NO_INLINE__)
    #define VECGEOM_INLINE inline __attribute__((always_inline))
  #else
    // Clang or forced inlining is disabled
    #define VECGEOM_INLINE inline
  #endif
#endif

#ifndef NULL
  #define NULL 0
#endif

// Allow constexpr variables and functions if possible
#ifdef VECGEOM_STD_CXX11
  #define VECGEOM_CONSTEXPR constexpr
  #define VECGEOM_CONSTEXPR_RETURN constexpr
#else
  #define VECGEOM_CONSTEXPR const
  #define VECGEOM_CONSTEXPR_RETURN
#endif

// Qualifier(s) of global constants
#ifndef VECGEOM_NVCC
  #define VECGEOM_GLOBAL constexpr
#else
  #define VECGEOM_GLOBAL static __constant__ const
#endif

namespace vecgeom {
#ifdef VECGEOM_FLOAT_PRECISION
typedef float Precision;
#else
typedef double Precision;
#endif
// namespace EInside {
// enum EInside {
//   kInside = 0,
//   kSurface = 1,
//   kOutside = 2
// };
// }
// typedef EInside::EInside Inside_t;
typedef int Inside_t;
}

namespace vecgeom_cuda {
typedef vecgeom::Precision Precision;
typedef vecgeom::Inside_t Inside_t;
}

namespace VECGEOM_NAMESPACE {

VECGEOM_GLOBAL int kAlignmentBoundary = 32;
VECGEOM_GLOBAL Precision kPi = 3.14159265358979323846;
VECGEOM_GLOBAL Precision kTwoPi = 2.*kPi;
VECGEOM_GLOBAL Precision kDegToRad = kPi/180.;
VECGEOM_GLOBAL Precision kRadToDeg = 180./kPi;
VECGEOM_GLOBAL Precision kInfinity =
#ifndef VECGEOM_NVCC
    std::numeric_limits<Precision>::infinity();
#else
    INFINITY;
#endif
VECGEOM_GLOBAL Precision kEpsilon =
#ifndef VECGEOM_NVCC
    std::numeric_limits<Precision>::epsilon();
#elif VECGEOM_FLOAT_PRECISION
    FLT_EPSILON;
#else
    DBL_EPSILON;
#endif
VECGEOM_GLOBAL Precision kTiny = 1e-30;
VECGEOM_GLOBAL Precision kTolerance = 1e-12;
VECGEOM_GLOBAL Precision kRadTolerance = 1e-12;
VECGEOM_GLOBAL Precision kAngTolerance = 1e-12;

VECGEOM_GLOBAL Precision kHalfTolerance = 0.5*kTolerance;
VECGEOM_GLOBAL Precision kToleranceSquared = kTolerance*kTolerance;

namespace EInside {
VECGEOM_GLOBAL VECGEOM_NAMESPACE::Inside_t kInside = 0;
VECGEOM_GLOBAL VECGEOM_NAMESPACE::Inside_t kSurface = 1;
VECGEOM_GLOBAL VECGEOM_NAMESPACE::Inside_t kOutside = 2;
}

// namespace EMatrix3DEntry {
// enum EMatrix3DEntry {
//   k00 = 0x001, k01 = 0x002, k02 = 0x004,
//   k10 = 0x008, k11 = 0x010, k12 = 0x020,
//   k20 = 0x040, k21 = 0x080, k22 = 0x100
// };
// }

typedef int RotationCode;
typedef int TranslationCode;
namespace rotation {
enum RotationId { kGeneric = -1, kDiagonal = 0x111, kIdentity = 0x200 };
}
namespace translation {
enum TranslationId { kGeneric = -1, kIdentity = 0 };
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Assert(const bool condition, char const *const message) {
#ifndef VECGEOM_NVCC
  if (!condition) {
    printf("Assertion failed: %s", message);
    abort();
  }
#else
  if (!condition) printf("Assertion failed: %s", message);
#endif
}

VECGEOM_CUDA_HEADER_BOTH
VECGEOM_INLINE
void Assert(const bool condition) {
  Assert(condition, "");
}

} // End global namespace

#endif // VECGEOM_BASE_GLOBAL_H_
