#ifndef VECGEOM_MATH_H
#define VECGEOM_MATH_H

#include <cmath>
#include <limits>
#include "VecGeom/base/Config.h"
#include "VecCore/Limits.h"

namespace vecgeom {

#ifdef VECGEOM_SINGLE_PRECISION
#define VECCORE_SINGLE_PRECISION
using Precision = float;
#else
using Precision = double;
#endif

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef __CUDA_ARCH__
#define VECGEOM_CONST static __constant__ const
#else
#define VECGEOM_CONST static constexpr
#endif

#ifdef VECGEOM_SINGLE_PRECISION
VECGEOM_CONST Precision kTolerance     = 1e-3;
VECGEOM_CONST Precision kInvTolerance  = 1e3;
VECGEOM_CONST Precision kPushTolerance = 1e-3;
VECGEOM_CONST Precision kSqrtTolerance = 3.1622777e-2;
VECGEOM_CONST Precision kAngTolerance  = 1e-2;
VECGEOM_CONST Precision kConeTolerance = 1e-3;
VECGEOM_CONST Precision kFarAway       = 1e5;
#else
VECGEOM_CONST Precision kTolerance     = 1e-9;
VECGEOM_CONST Precision kInvTolerance  = 1e9;
VECGEOM_CONST Precision kPushTolerance = 1e-6;
VECGEOM_CONST Precision kSqrtTolerance = 3.1622777e-5;
VECGEOM_CONST Precision kAngTolerance  = 1e-9;
VECGEOM_CONST Precision kConeTolerance = 1e-7;
VECGEOM_CONST Precision kFarAway       = 1e10;
#endif

// Tolerace distance constant specializations
template <typename Real_t>
constexpr Real_t kToleranceDist = Real_t(0);
template <>
inline constexpr double kToleranceDist<double> = double(1e-9);
template <>
inline constexpr float kToleranceDist<float> = float(1e-3);

template <typename Real_t>
constexpr Real_t kToleranceStrict = Real_t(0);
template <>
inline constexpr double kToleranceStrict<double> = double(1e-9);
template <>
inline constexpr float kToleranceStrict<float> = float(1e-6);

template <typename Real_t>
constexpr Real_t kToleranceDistSquared = Real_t(0);
template <>
inline constexpr double kToleranceDistSquared<double> = kToleranceDist<double> * kToleranceDist<double>;
template <>
inline constexpr float kToleranceDistSquared<float> = kToleranceDist<float> * kToleranceDist<float>;

template <typename Real_t>
constexpr Real_t kToleranceCone = Real_t(0);
template <>
inline constexpr double kToleranceCone<double> = double(1e-7);
template <>
inline constexpr float kToleranceCone<float> = float(1e-3);

template <typename Real_t>
constexpr Real_t kToleranceArb4 = Real_t(0);
template <>
inline constexpr double kToleranceArb4<double> = double(1e-6);
template <>
inline constexpr float kToleranceArb4<float> = float(1e-3);

#ifdef VECGEOM_BVH_SINGLE
constexpr double kToleranceBVH = double(1e-4);
#else
constexpr float kToleranceBVH = float(1e-9);
#endif

using namespace vecCore::math;

VECGEOM_CONST Precision kAvogadro = 6.02214085774e23;
VECGEOM_CONST Precision kEpsilon  = std::numeric_limits<Precision>::epsilon();
template <typename Real_t>
constexpr Real_t kEpsilonT        = std::numeric_limits<Real_t>::epsilon();
VECGEOM_CONST double kInfinityDbl = std::numeric_limits<double>::infinity();
// a function to estimate ULP *unit in the last place for a number
// Compute ULP of a given number x (templated on precision type)
template <typename T>
VECCORE_ATT_HOST_DEVICE T ULP(T x)
{
  static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
  // Use nextafter to find the next representable value greater than x
  T next = std::nextafter(x, vecCore::NumericLimits<T>::Infinity());
  return vecCore::math::Abs(next - x);
}

// a special constant to indicate a "miss" length
VECGEOM_CONST Precision kInfLength        = vecCore::NumericLimits<Precision>::Max();
VECGEOM_CONST Precision kMaximum          = vecCore::NumericLimits<Precision>::Max();
VECGEOM_CONST int kMaximumInt             = vecCore::NumericLimits<int>::Max();
VECGEOM_CONST Precision kMinimum          = vecCore::NumericLimits<Precision>::Min();
VECGEOM_CONST Precision kPi               = 3.14159265358979323846;
VECGEOM_CONST Precision kHalfPi           = 0.5 * kPi;
VECGEOM_CONST Precision kTwoPi            = 2. * kPi;
VECGEOM_CONST Precision kTwoPiInv         = 1. / kTwoPi;
VECGEOM_CONST Precision kDegToRad         = kPi / 180.;
VECGEOM_CONST Precision kRadToDeg         = 180. / kPi;
VECGEOM_CONST Precision kRadTolerance     = 1e-9;
VECGEOM_CONST Precision kTiny             = 1e-30;
VECGEOM_CONST Precision kHalfTolerance    = 0.5 * kTolerance;
VECGEOM_CONST Precision kToleranceSquared = kTolerance * kTolerance;

template <typename T>
struct Tiny {
  static constexpr T kValue = 1.e-30;
};

template <template <typename, typename> class ImplementationType, typename T, typename Q>
struct Tiny<ImplementationType<T, Q>> {
  static constexpr typename ImplementationType<T, Q>::value_type kValue = 1.e-30;
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
