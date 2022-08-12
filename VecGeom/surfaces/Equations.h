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
  Real_t phalf {0};   // we don't need p itself when solving the equation, hence we store just p/2
  Real_t q {0};       // we do need q
};

template <typename Real_t>
void CylinderEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir,
                Real_t radius, QuadraticCoef<Real_t> &coef)
{
  Real_t radius2  = radius * radius;
  Real_t rsq      = point.x() * point.x() + point.y() * point.y();
  Real_t rdotn    = point.x() * dir.x() + point.y() * dir.y();
  Real_t invnsq   = Real_t(1.) / vecgeom::NonZero(Real_t(1.) - dir.z() * dir.z());
 
  coef.phalf      = invnsq * rdotn;
  coef.q          = invnsq * (rsq - radius2);
}

template <typename Real_t>
void ConeEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir,
            Real_t radius, Real_t slope, QuadraticCoef<Real_t> &coef)
{  
}

template <typename Real_t>
void SphereEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir,
              Real_t radius, QuadraticCoef<Real_t> &coef)
{  
}

template <typename Real_t>
void QuadraticSolver(QuadraticCoef<Real_t> const &coef, Real_t* roots, int &numroots)
{
  Real_t delta = coef.phalf * coef.phalf - coef.q;

  if (std::abs(delta) < vecgeom::kTolerance) {
    roots[0] = roots[1] = -coef.phalf;
    numroots = 1;
    return;
  }
  else if (delta < Real_t(0.)) {
    numroots = -1;
    return;
  }

  delta = std::sqrt(delta);

  roots[1] = -coef.phalf + delta;
  // If the greater solution is less than zero, we have no positive solutions
  if (roots[1] < -vecgeom::kTolerance) {
    numroots = 0;
    return;
  }
  roots[0] = -coef.phalf - delta;
  // If smaller solution is less than zero, we have just one positive solution
  if (roots[0] < -vecgeom::kTolerance) {
    roots[0] = roots[1];
    numroots = 1;
    return;
  }
  numroots = 2;
}

} // namespace vgbrep
#endif
