/// @file global.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_GLOBAL_H_
#define VECGEOM_BASE_GLOBAL_H_

#include <cmath>
#include <limits>

#if (defined(__CUDACC__) || defined(__NVCC__))
  #define VECGEOM_NVCC
  #define VECGEOM_NAMESPACE vecgeom_cuda
  #define VECGEOM_CUDA_HEADER_DEVICE __device__
  #define VECGEOM_CUDA_HEADER_HOST __host__
  #define VECGEOM_CUDA_HEADER_BOTH __host__ __device__
  #define VECGEOM_CUDA_HEADER_GLOBAL __global__
#else // Not compiling with NVCC
  #define VECGEOM_CUDA_HEADER_DEVICE
  #define VECGEOM_CUDA_HEADER_HOST
  #define VECGEOM_CUDA_HEADER_BOTH
  #define VECGEOM_CUDA_HEADER_GLOBAL
  #ifdef VECGEOM_CUDA
    #define VECGEOM_CUDA_INTERFACE
  #endif
#endif

#ifdef VECGEOM_NVCC
  #undef VECGEOM_VC
  #undef VECGEOM_VC_ACCELERATION
  #undef VECGEOM_CILK
  #undef VECGEOM_ROOT
  #undef VECGEOM_USOLIDS
  #undef VECGEOM_BENCHMARK
#else
  #define VECGEOM_STD_CXX11
  #define VECGEOM_NAMESPACE vecgeom
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

#ifndef VECGEOM_NVCC
  #define VECGEOM_CONSTEXPR constexpr
#else
  #define VECGEOM_CONSTEXPR static __constant__ const
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

VECGEOM_CONSTEXPR int kAlignmentBoundary = 32;
VECGEOM_CONSTEXPR Precision kPi = 3.14159265358979323846;
VECGEOM_CONSTEXPR Precision kDegToRad = kPi/180.;
VECGEOM_CONSTEXPR Precision kRadToDeg = 180./kPi;
VECGEOM_CONSTEXPR Precision kInfinity =
#ifndef VECGEOM_NVCC
    std::numeric_limits<Precision>::infinity();
#else
    INFINITY;
#endif
VECGEOM_CONSTEXPR Precision kTiny = 1e-30;
VECGEOM_CONSTEXPR Precision kTolerance = 1e-12;
VECGEOM_CONSTEXPR Precision kHalfTolerance = 0.5*kTolerance;

namespace EInside {
VECGEOM_CONSTEXPR VECGEOM_NAMESPACE::Inside_t kInside = 0;
VECGEOM_CONSTEXPR VECGEOM_NAMESPACE::Inside_t kSurface = 1;
VECGEOM_CONSTEXPR VECGEOM_NAMESPACE::Inside_t kOutside = 2;
}

template <typename Type>
class Vector3D;

template <typename Type>
class SOA3D;

template <typename Type>
class AOS3D;

template <typename Type>
class Container;

template <typename Type>
class Vector;

template <typename Type>
class Array;

class LogicalVolume;

class VPlacedVolume;

class VUnplacedVolume;

class UnplacedBox;

class PlacedBox;

class Transformation3D;

class GeoManager;

#ifdef VECGEOM_CUDA_INTERFACE
class CudaManager;
#endif

namespace matrix3d_entry {
enum Matrix3DEntry {
  k00 = 0x001, k01 = 0x002, k02 = 0x004,
  k10 = 0x008, k11 = 0x010, k12 = 0x020,
  k20 = 0x040, k21 = 0x080, k22 = 0x100
};
}

typedef int RotationCode;
typedef int TranslationCode;
namespace rotation {
enum RotationId { kGeneric = -1, kDiagonal = 0x111, kIdentity = 0x200 };
}
namespace translation {
enum TranslationId { kGeneric = -1, kIdentity = 0 };
}

} // End global namespace



#endif // VECGEOM_BASE_GLOBAL_H_
