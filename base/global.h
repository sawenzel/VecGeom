/**
 * @file global.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BASE_GLOBAL_H_
#define VECGEOM_BASE_GLOBAL_H_

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

#include <cmath>

#ifndef NULL
  #define NULL 0
#endif

namespace vecgeom {
#ifdef VECGEOM_FLOAT_PRECISION
typedef float Precision;
#else
typedef double Precision;
#endif
}

namespace vecgeom_cuda {
typedef vecgeom::Precision Precision;
}

namespace VECGEOM_NAMESPACE {

const int kAlignmentBoundary = 32;
const Precision kDegToRad = M_PI/180.;
const Precision kRadToDeg = 180./M_PI;
const Precision kInfinity = INFINITY;
const Precision kTiny = 1e-20;
const Precision kTolerance = 1e-12;

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

typedef VPlacedVolume VUSolid;

class VUnplacedVolume;

class UnplacedBox;

class PlacedBox;

class TransformationMatrix;

class GeoManager;

#ifdef VECGEOM_CUDA_INTERFACE
class CudaManager;
#endif

namespace entry {
enum Entry {
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
enum TranslationId { kGeneric = -1, kOrigin = 0 };
}

} // End global namespace

#ifdef VECGEOM_ROOT
class TGeoShape;
class TGeoBBox;
class TGeoNode;
class TGeoMatrix;
class TGeoVolume;
#endif

#ifdef VECGEOM_USOLIDS
class VUSolid;
class UBox;
#endif

#endif // VECGEOM_BASE_GLOBAL_H_
