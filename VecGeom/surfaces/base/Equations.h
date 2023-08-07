#ifndef VECGEOM_SURFACE_EQUATIONS_H_
#define VECGEOM_SURFACE_EQUATIONS_H_

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/base/Math.h>

namespace vgbrep {

template <typename Real_t>
using Vector3D = vecgeom::Vector3D<Real_t>;

///< Second order equation coefficients, as in: <x * x + 2*phalf * x + q = 0>
template <typename Real_t>
struct QuadraticCoef {
  Real_t phalf{0}; // we don't need p itself when solving the equation, hence we store just p/2
  Real_t q{0};     // we do need q
};

/// @brief Fill equation for ray-cylinder intersections
/// @param point Starting point in local coordinates
/// @param dir Direction in local coordinates
/// @param radius Cylinder radius
/// @return coef Quadratic equation giving ray-cylinder intersections
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void CylinderEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Real_t radius,
                                        QuadraticCoef<Real_t> &coef)
{
  // TODO: The cylyndrical and cone equations can be merged
  Real_t rsq    = point.Perp2();
  Real_t rdotn  = point.x() * dir.x() + point.y() * dir.y();
  Real_t invnsq = Real_t(1.) / vecgeom::NonZero(dir.Perp2());

  coef.phalf = invnsq * rdotn;
  coef.q     = invnsq * (rsq - radius * radius);
}

/// @brief Fill equation for ray-cone intersections
/// @param point Starting point in local coordinates
/// @param dir Direction in local coordinates
/// @param radius Cone radius at Z = 0
/// @param slope Cone surface slope
/// @return coef Quadratic equation giving ray-cone intersections
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void ConeEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Real_t radius,
                                    Real_t slope, QuadraticCoef<Real_t> &coef)
{
  Real_t rz     = radius + point[2] * slope;
  Real_t rsq    = point.Perp2();
  Real_t rdotn  = point.x() * dir.x() + point.y() * dir.y() - slope * rz * dir.z();
  Real_t invnsq = Real_t(1.) / vecgeom::NonZero(dir.Perp2() - slope * slope * dir.z() * dir.z());

  coef.phalf = invnsq * rdotn;
  coef.q     = invnsq * (rsq - rz * rz);
}

/// @brief Fill equation for ray-sphere intersections
/// @param point Starting point in local coordinates
/// @param dir Direction in local coordinates
/// @param radius Sphere radius
/// @return coef Quadratic equation giving ray-sphere intersections
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void SphereEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir, Real_t radius,
                                      QuadraticCoef<Real_t> &coef)
{
  Real_t rsq = point.Mag2();
  coef.phalf = point.Dot(dir);
  coef.q     = rsq - radius * radius;
}

/// @brief Solver for quadratic equations
/// @tparam Real_t Floating point type
/// @param coef Quadratic equation coefficients
/// @param roots Equation roots
/// @param numroots Number of roots grater than -kTolerance
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE void QuadraticSolver(QuadraticCoef<Real_t> const &coef, Real_t *roots, int &numroots)
{
  numroots     = 0;
  Real_t delta = coef.phalf * coef.phalf - coef.q;
  if (delta < Real_t(0)) return;

  delta           = std::sqrt(delta);
  roots[numroots] = -coef.phalf - delta;
  if (roots[numroots] > Real_t(-vecgeom::kTolerance)) numroots++;

  roots[numroots] = -coef.phalf + delta;
  if (roots[numroots] > Real_t(-vecgeom::kTolerance)) numroots++;
}

} // namespace vgbrep
#endif
