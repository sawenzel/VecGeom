/// \file Global.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_GLOBAL_H_
#define VECGEOM_BASE_GLOBAL_H_

#if __cplusplus < 201103L
#error VecGeom requires compiler and library support for the ISO C++ 2011 standard.
#endif

#include "VecGeom/base/Config.h"

#ifdef VECGEOM_FLOAT_PRECISION
#define VECCORE_SINGLE_PRECISION
using Precision = float;
#else
using Precision = double;
#endif

#include <VecCore/VecCore>

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Math.h"
#include <type_traits>

using uint       = unsigned int;
using NavIndex_t = unsigned int;

#define VECGEOM

#ifdef __INTEL_COMPILER
// Compiling with icc
#define VECGEOM_INTEL
#define VECGEOM_FORCE_INLINE inline
#ifndef VECCORE_CUDA
#define VECGEOM_ALIGNED __attribute__((aligned(64)))
#endif
#else
#if (defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)) && !defined(__NO_INLINE__) && \
    !defined(VECGEOM_NOINLINE)
#define VECGEOM_FORCE_INLINE inline __attribute__((always_inline))
#ifndef VECCORE_CUDA
#define VECGEOM_ALIGNED __attribute__((aligned(64)))
#endif
#else
// Clang or forced inlining is disabled ( by falling back to compiler decision )
#define VECGEOM_FORCE_INLINE inline
#ifndef VECCORE_CUDA
#define VECGEOM_ALIGNED
#endif
#endif
#endif

// Allow constexpr variables and functions if possible
#define VECGEOM_CONSTEXPR constexpr
#define VECGEOM_CONSTEXPR_RETURN constexpr

// Qualifier(s) of global constants
#ifdef VECCORE_CUDA_DEVICE_COMPILATION
// constexpr not supported on device in CUDA 6.5
#define VECGEOM_GLOBAL static __constant__ const
#define VECGEOM_CLASS_GLOBAL static const
#else
#define VECGEOM_GLOBAL static constexpr
#define VECGEOM_CLASS_GLOBAL static constexpr
#endif

namespace vecgeom {

enum class ESolidType : char {
  boolean = 0,
  box,
  cone,
  coaxialcones,
  cuttube,
  ellipsoid,
  ellipticalcone,
  ellipticaltube,
  extruded,
  gentrap,
  genericpolycone,
  hyperboloid,
  multiunion,
  orb,
  paraboloid,
  parallelepiped,
  polycone,
  polyhedron,
  sextruded,
  scaled,
  sphere,
  tessellated,
  tetrahedron,
  torus,
  trapezoid,
  trd,
  tube,
  nosolid
};

inline namespace VECGEOM_IMPL_NAMESPACE {
enum EnumInside {
  eInside  = 1, /* for USOLID compatibility */
  kInside  = eInside,
  eSurface = 2,
  kSurface = eSurface,
  eOutside = 3,
  kOutside = eOutside,
};

using Inside_t = int;

VECGEOM_GLOBAL int kAlignmentBoundary = 32;

namespace EInside {
VECGEOM_GLOBAL vecgeom::Inside_t kInside  = 1;
VECGEOM_GLOBAL vecgeom::Inside_t kSurface = 2;
VECGEOM_GLOBAL vecgeom::Inside_t kOutside = 3;
} // namespace EInside

namespace details {
template <typename DataType, typename Target>
struct UseIfSameType {
  VECCORE_ATT_HOST_DEVICE
  static Target const *Get(DataType *) { return nullptr; }
};
template <typename DataType>
struct UseIfSameType<DataType, DataType> {
  VECCORE_ATT_HOST_DEVICE
  static DataType const *Get(DataType *ptr) { return ptr; }
};
} // namespace details

// some static MACROS
#define VECGEOM_MAXDAUGHTERS 2000 // macro mainly used to allocated static (stack) arrays/workspaces
#define VECGEOM_MAXFACETS 20000   // macro mainly used to allocated static (stack hybrid navigator arrays/workspaces

// choosing the Vector and Scalar backends
// trying to set some sort of default scalar and vector backend
#if defined(VECGEOM_VC) && !defined(VECCORE_CUDA)
using VectorBackend = vecCore::backend::VcVectorT<Precision>;
#else
using VectorBackend = vecCore::backend::ScalarT<Precision>;
#endif
using ScalarBackend = vecCore::backend::ScalarT<Precision>;

// anonymous namespace around purely local helper functions
namespace {
// helper code for the MaskedAssignFunc macro
template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool ToBool(T /* mask */)
{
  return false;
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool ToBool<bool>(bool mask)
{
  return mask;
#pragma GCC diagnostic pop
}
} // end anonymous namespace

// define a macro for MaskedAssignFunc which should be
// used in case the third argument are expensive expressions
// (such as function calls or arithmetic operations)
// FIXME: move this to VecCore
#define vecCore__MaskedAssignFunc(Dest, Mask, FuncCallExpr)                                 \
  {                                                                                         \
    if (vecCore::VectorSize<typename std::remove_reference<decltype(Dest)>::type>() == 1) { \
      if (vecgeom::ToBool(Mask)) Dest = FuncCallExpr;                                       \
    } else {                                                                                \
      vecCore::MaskedAssign(Dest, Mask, FuncCallExpr);                                      \
    }                                                                                       \
  }

} // namespace VECGEOM_IMPL_NAMESPACE

// defining an infinite length constant
template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T InfinityLength() noexcept
{
  return vecCore::NumericLimits<T>::Max();
}

// is this in VecCore??
template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T NonZeroAbs(T const &x)
{
  return Abs(x) + T(1.0e-30);
}

template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T NonZero(T const &x)
{
  return x + CopySign(T(1.0e-30), x);
}

} // namespace vecgeom

#endif // VECGEOM_BASE_GLOBAL_H_
