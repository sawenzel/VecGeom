// Shared ApproxEqual helpers for VecGeom tests.

#ifndef VECGEOM_TEST_VECGEOMTEST_APPROXEQUAL_HH
#define VECGEOM_TEST_VECGEOMTEST_APPROXEQUAL_HH

#include <cmath>
#include <type_traits>

namespace vecgeom {
namespace test {

constexpr double kApproxEqualTolerance    = 1E-6;
constexpr float kApproxEqualToleranceFlt  = 1E-3f;
constexpr double kInfinityComparisonLimit = 1.0e+100;

template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline bool ApproxEqual(const T &x, const T &y)
{
  if (x == y) return true;

  const auto diff  = std::fabs(x - y);
  const auto abs_x = std::fabs(x);
  const auto abs_y = std::fabs(y);
  const auto tol   = std::is_same<T, float>::value ? static_cast<T>(kApproxEqualToleranceFlt)
                                                   : static_cast<T>(kApproxEqualTolerance);

  if (x * y == static_cast<T>(0.0)) return diff < tol;

  if (abs_x > static_cast<T>(kInfinityComparisonLimit) || abs_y > static_cast<T>(kInfinityComparisonLimit)) {
    return (x * y > static_cast<T>(0.0)) && abs_x > static_cast<T>(kInfinityComparisonLimit) &&
           abs_y > static_cast<T>(kInfinityComparisonLimit);
  }

  return diff / (abs_x + abs_y) < tol;
}

template <typename Vec_t>
inline auto ApproxEqual(const Vec_t &check, const Vec_t &target)
    -> decltype(check.x(), check.y(), check.z(), target.x(), target.y(), target.z(), bool())
{
  return ApproxEqual(check.x(), target.x()) && ApproxEqual(check.y(), target.y()) &&
         ApproxEqual(check.z(), target.z());
}

} // namespace test
} // namespace vecgeom

#endif
