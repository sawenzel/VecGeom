/// @file CoaxialConesStruct.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_COAXIALCONESSTRUCT_H_
#define VECGEOM_VOLUMES_COAXIALCONESSTRUCT_H_
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/volumes/ConeStruct.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T = double>
struct CoaxialConesStruct {

  // CoaxialCone parameters
  unsigned int fNumOfCones;
  T fDz;
  T fSPhi;
  T fDPhi;
  Vector<T> fRmin1Vect;
  Vector<T> fRmax1Vect;
  Vector<T> fRmin2Vect;
  Vector<T> fRmax2Vect;

  /*
  ** A function to Print the sequence of Coaxial cones,
  ** Useful for debugging the Coaxial cones
  */
  void Print() const
  {

    for (unsigned int i = 0; i < fConeStructVector.size(); i++) {
      std::cerr << std::endl << "====== Index of cone struct : " << i << " ======" << std::endl;
      fConeStructVector[i]->Print();
      std::cerr << std::endl;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  CoaxialConesStruct() {}

  VECCORE_ATT_HOST_DEVICE
  CoaxialConesStruct(unsigned int numOfCones, T *rmin1Vect, T *rmax1Vect, T *rmin2Vect, T *rmax2Vect, T dz, T sphi,
                     T dphi)
      : fNumOfCones(numOfCones), fDz(dz), fSPhi(sphi), fDPhi(dphi)
  {

    for (unsigned int i = 0; i < fNumOfCones; i++) {
      fRmin1Vect.push_back(rmin1Vect[i]);
      fRmax1Vect.push_back(rmax1Vect[i]);
      fRmin2Vect.push_back(rmin2Vect[i]);
      fRmax2Vect.push_back(rmax2Vect[i]);
    }

    for (unsigned int i = 0; i < fNumOfCones; i++) {
      fConeStructVector.push_back(
          new ConeStruct<T>(fRmin1Vect[i], fRmax1Vect[i], fRmin2Vect[i], fRmax2Vect[i], dz, sphi, dphi));
      if (i == 0) {
        fMinR = fRmin1Vect[i] < fRmin2Vect[i] ? fRmin1Vect[i] : fRmin2Vect[i];
      }

      if (i == fNumOfCones - 1) {
        fMaxR = fRmax1Vect[i] > fRmax2Vect[i] ? fRmax1Vect[i] : fRmax2Vect[i];
      }
    }
  }

  VECCORE_ATT_HOST_DEVICE
  CoaxialConesStruct(Vector<Precision> rmin1Vect, Vector<Precision> rmax1Vect, Vector<Precision> rmin2Vect,
                     Vector<Precision> rmax2Vect, T dz, T sphi, T dphi)
      : fNumOfCones(rmin1Vect.size()), fDz(dz), fSPhi(sphi), fDPhi(dphi), fRmin1Vect(rmin1Vect), fRmax1Vect(rmax1Vect),
        fRmin2Vect(rmin2Vect), fRmax2Vect(rmax2Vect)
  {

    for (unsigned int i = 0; i < fNumOfCones; i++) {
      fConeStructVector.push_back(
          new ConeStruct<T>(fRmin1Vect[i], fRmax1Vect[i], fRmin2Vect[i], fRmax2Vect[i], dz, sphi, dphi));
      if (i == 0) {
        fMinR = fRmin1Vect[i] < fRmin2Vect[i] ? fRmin1Vect[i] : fRmin2Vect[i];
      }

      if (i == fNumOfCones - 1) {
        fMaxR = fRmax1Vect[i] > fRmax2Vect[i] ? fRmax1Vect[i] : fRmax2Vect[i];
      }
    }
  }

  // Vector of Cones
  Vector<ConeStruct<T> *> fConeStructVector;

  T fSurfaceArea; // area of the surface
  T fCubicVolume; // volume
  T fMaxR;
  T fMinR;

  // Precalculated cached values
  VECCORE_ATT_HOST_DEVICE
  Precision Capacity()
  {
    Precision volume = 0.;
    for (unsigned int i = 0; i < fConeStructVector.size(); i++) {
      volume += fConeStructVector[i]->Capacity();
    }
    return volume;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision ConicalSurfaceArea()
  {
    Precision conicalSurfaceArea = 0.;
    Precision mmin, mmax, dmin, dmax;

    for (unsigned int i = 0; i < fConeStructVector.size(); i++) {

      mmin = (fConeStructVector[i]->fRmin1 + fConeStructVector[i]->fRmin2) * 0.5;
      mmax = (fConeStructVector[i]->fRmax1 + fConeStructVector[i]->fRmax2) * 0.5;
      dmin = (fConeStructVector[i]->fRmin2 - fConeStructVector[i]->fRmin1);
      dmax = (fConeStructVector[i]->fRmax2 - fConeStructVector[i]->fRmax1);

      conicalSurfaceArea +=
          fConeStructVector[i]->fDPhi *
          (mmin * vecCore::math::Sqrt(dmin * dmin + 4 * fConeStructVector[i]->fDz * fConeStructVector[i]->fDz) +
           mmax * vecCore::math::Sqrt(dmax * dmax + 4 * fConeStructVector[i]->fDz * fConeStructVector[i]->fDz));
    }
    return conicalSurfaceArea;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision SurfaceAreaLowerZPlanes(int index)
  {
    return fConeStructVector[index]->fDPhi * 0.5 *
           (fConeStructVector[index]->fRmax1 * fConeStructVector[index]->fRmax1 -
            fConeStructVector[index]->fRmin1 * fConeStructVector[index]->fRmin1);
  }

  VECCORE_ATT_HOST_DEVICE
  Precision SurfaceAreaUpperZPlanes(int index)
  {
    return fConeStructVector[index]->fDPhi * 0.5 *
           (fConeStructVector[index]->fRmax2 * fConeStructVector[index]->fRmax2 -
            fConeStructVector[index]->fRmin2 * fConeStructVector[index]->fRmin2);
  }

  VECCORE_ATT_HOST_DEVICE
  Precision SurfaceAreaOfZPlanes()
  {
    Precision surfaceAreaOfZPlanes = 0.;
    for (unsigned int i = 0; i < fConeStructVector.size(); i++) {
      surfaceAreaOfZPlanes = SurfaceAreaLowerZPlanes(i) + SurfaceAreaUpperZPlanes(i);
    }
    return surfaceAreaOfZPlanes;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() { return (ConicalSurfaceArea() + SurfaceAreaOfZPlanes()); }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &p, Vector3D<Precision> &norm) const
  {

    norm.Set(0.);
    bool valid = false;
    for (unsigned int i = 0; i < fConeStructVector.size(); i++) {
      bool validNormal = false;
      Vector3D<Precision> normal(0., 0., 0.);
      validNormal = fConeStructVector[i]->Normal(p, normal);
      if (validNormal) {
        norm += normal;
      }
      valid |= validNormal;
    }
    if (valid) {
      norm.Normalize();
    } else {
      norm.Set(0., 0., 1.);
    }
    return valid;
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
