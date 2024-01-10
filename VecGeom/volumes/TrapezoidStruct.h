/// @file: TrapezoidStruct.h
/// @author Guilherme Lima (lima@fnal.gov)
//
//  2016-07-22 Guilherme Lima  Created
//

#ifndef VECGEOM_VOLUMES_TRAPEZOIDSTRUCT_H_
#define VECGEOM_VOLUMES_TRAPEZOIDSTRUCT_H_
#include "VecGeom/base/Global.h"
#include "VecGeom/base/PlaneShell.h"
#include "VecCore/VecMath.h"

namespace vecgeom {

// using namespace vecCore::math;

inline namespace VECGEOM_IMPL_NAMESPACE {

/*
 * A Trapezoid struct to encapsulate the parameters and some other cached values
 * related to Trapezoid that are required in Implementation
 */
template <typename T = double>
struct TrapezoidStruct {

  struct TrapSidePlane {
    Precision fA, fB, fC, fD;
    // Plane equation: Ax+By+Cz+D=0, where
    // normal unit vector nvec=(A,B,C)  and offset=D is the distance from origin to plane

    VECCORE_ATT_HOST_DEVICE
    TrapSidePlane() : fA(0.0), fB(0.0), fC(0.0), fD(0.0) {}

    VECCORE_ATT_HOST_DEVICE
    TrapSidePlane(Precision a, Precision b, Precision c, Precision d) : fA(a), fB(b), fC(c), fD(d) {}

    VECCORE_ATT_HOST_DEVICE
    TrapSidePlane(TrapSidePlane const &oth) : fA(oth.fA), fB(oth.fB), fC(oth.fC), fD(oth.fD) {}
  };

  T fDz;
  T fTheta;
  T fPhi;
  T fDy1;
  T fDx1;
  T fDx2;
  T fTanAlpha1;
  T fDy2;
  T fDx3;
  T fDx4;
  T fTanAlpha2;

  // Values computed from parameters, to be cached
  T fTthetaCphi;
  T fTthetaSphi;

#ifndef VECGEOM_PLANESHELL_DISABLE
  typedef PlaneShell<4, Precision> Planes;
  Planes fPlanes;
#else
  TrapSidePlane fPlanes[4];
#endif

  T sideAreas[6]; // including z-planes
  Vector3D<T> normals[6];

public:
  /// \brief Constructors
  /// @{
  // \brief General constructor.  All other constructors should delegate to it
  VECCORE_ATT_HOST_DEVICE
  TrapezoidStruct(const T pDz, const T pTheta, const T pPhi, const T pDy1, const T pDx1, const T pDx2,
                  const T pTanAlpha1, const T pDy2, const T pDx3, const T pDx4, const T pTanAlpha2)
      : fDz(pDz), fTheta(pTheta), fPhi(pPhi), fDy1(pDy1), fDx1(pDx1), fDx2(pDx2), fTanAlpha1(pTanAlpha1), fDy2(pDy2),
        fDx3(pDx3), fDx4(pDx4), fTanAlpha2(pTanAlpha2)
  {
    CalculateCached();
  }

  /// \brief Constructor for a "default" trapezoid whose parameters are to be set later
  VECCORE_ATT_HOST_DEVICE
  TrapezoidStruct() : TrapezoidStruct(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.) {}
  /// @}

  /// \brief Destructor
  VECCORE_ATT_HOST_DEVICE
  virtual ~TrapezoidStruct() {}

  VECCORE_ATT_HOST_DEVICE
  void CalculateCached()
  {
    fTthetaCphi = vecCore::math::Tan(fTheta) * vecCore::math::Cos(fPhi);
    fTthetaSphi = vecCore::math::Tan(fTheta) * vecCore::math::Sin(fPhi);
    // Recalculate theta and phi for consistency with Geant4 
    fTheta = vecCore::math::ATan(vecCore::math::Abs(vecCore::math::Tan(fTheta)));
    fPhi = vecCore::math::ATan2(fTthetaSphi, fTthetaCphi);
  }

public:
#ifndef VECGEOM_PLANESHELL_DISABLE

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Planes const *GetPlanes() const { return &fPlanes; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TrapSidePlane GetPlane(unsigned int i) const
  {
    return TrapSidePlane(fPlanes.fA[i], fPlanes.fB[i], fPlanes.fC[i], fPlanes.fD[i]);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetPlane(unsigned int i, Precision a, Precision b, Precision c, Precision d)
  {
    fPlanes.fA[i] = a;
    fPlanes.fB[i] = b;
    fPlanes.fC[i] = c;
    fPlanes.fD[i] = d;
  }

#else

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TrapSidePlane const *GetPlanes() const { return fPlanes; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  TrapSidePlane GetPlane(unsigned int i) const { return fPlanes[i]; }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetPlane(unsigned int i, Precision a, Precision b, Precision c, Precision d)
  {
    fPlanes[i].fA = a;
    fPlanes[i].fB = b;
    fPlanes[i].fC = c;
    fPlanes[i].fD = d;
  }
#endif
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
