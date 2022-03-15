#ifndef VECGEOM_SURFACE_EQUATIONS_H_
#define VECGEOM_SURFACE_EQUATIONS_H_

#include <VecGeom/base/Vector3D.h>

namespace vgbrep {

template <typename Real_t>
using Vector3D = vecgeom::Vector3D<Real_t>;

///< Second order equation coefficients, as in: <x * x + p * x + q = 0>
template <typename Real_t>
struct QuadraticCoef {
  Real_t p {0};
  Real_t q {0};
};

///< Sorted quadratic equation roots
template <typename Real_t>
struct QuadraticRoots {
  Real_t xmin {0};
  Real_t xmax {0};
  int numroots;
};

template <typename Real_t>
void CylinderEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir,
                Real_t radius, QuadraticCoef<Real_t> const &coef)
{
}

template <typename Real_t>
void ConeEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir,
            Real_t radius, Real_t slope, QuadraticCoef<Real_t> const &coef)
{  
}

template <typename Real_t>
void SphereEq(Vector3D<Real_t> const &point, Vector3D<Real_t> const &dir,
              Real_t radius, QuadraticCoef<Real_t> const &coef)
{  
}

template <typename Real_t>
void QuadraticSolver(QuadraticCoef<Real_t> const &coef, QuadraticRoots<Real_t> &roots)
{
}

} // namespace vgbrep
#endif
