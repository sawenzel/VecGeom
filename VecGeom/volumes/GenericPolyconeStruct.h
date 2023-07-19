/// @file GenericPolyconeStruct.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_GENERICPOLYCONESTRUCT_H_
#define VECGEOM_VOLUMES_GENERICPOLYCONESTRUCT_H_
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/volumes/CoaxialConesStruct.h"
#include "VecGeom/volumes/GenericPolyconeSection.h"
namespace vecgeom {

// VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(struct, GenericPolyconeStruct, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T = double>
struct GenericPolyconeStruct {

  // Data members to store
  Vector<Vector<T>> fVectOfRmin1Vect;
  Vector<Vector<T>> fVectOfRmax1Vect;
  Vector<Vector<T>> fVectOfRmin2Vect;
  Vector<Vector<T>> fVectOfRmax2Vect;
  Vector<T> fVectOfDz;
  T fSPhi;
  T fDPhi;

  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeStruct() {}

  VECCORE_ATT_HOST_DEVICE
  // Special function NOT to be exposed to user, and will be used by constructor
  void Set(Vector<Vector<Precision>> vectOfRmin1Vect, Vector<Vector<Precision>> vectOfRmax1Vect,
           Vector<Vector<Precision>> vectOfRmin2Vect, Vector<Vector<Precision>> vectOfRmax2Vect, Vector<Precision> zS,
           Precision sPhi, Precision dPhi)
  {
    fSPhi = sPhi;
    fDPhi = dPhi;
    for (unsigned int i = 0; i < vectOfRmin1Vect.size(); i++) {

      // Creating and filling the GenericPolyconeSection
      GenericPolyconeSection section;
      // Precision shift                             = zS[i] + 0.5 * (zS[i + 1] - zS[i]);
      Precision dz                                = 0.5 * (zS[i + 1] - zS[i]);
      Precision shift                             = zS[i] + dz;
      CoaxialConesStruct<Precision> *coaxialCones = new CoaxialConesStruct<Precision>(
          vectOfRmin1Vect[i], vectOfRmax1Vect[i], vectOfRmin2Vect[i], vectOfRmax2Vect[i], dz, fSPhi, fDPhi);
      fCubicVolume += coaxialCones->Capacity();
      section.fShift        = shift;
      section.fCoaxialCones = coaxialCones;

      // Inserting the section just created above
      fSections.push_back(section);
    }
    fZs          = zS;
    fCubicVolume = Capacity();
  }

  // Vector of GenericPolyconeSection
  Vector<GenericPolyconeSection> fSections;

  /* Variables to be Cached
   * 1) All the Z Planes fZs
   */

  Vector<Precision> fZs;

  T fSurfaceArea; // area of the surface
  T fCubicVolume; // volume

  // Some Helper function to calculate SurfaceArea
#if (0)
  Precision ConicalSurfaceArea()
  {
    Precision conicalSurfaceArea = 0.;
    for (unsigned int i = 0; i < fSections.size(); i++) {
      conicalSurfaceArea += fSections[i].fCoaxialCones->ConicalSurfaceArea();
    }
    return conicalSurfaceArea;
  }

  Precision SurfaceAreaOfZPlanes()
  {
    Precision surfaceAreaOfZPlanes = 0.;
    if (fSections.size() == 1) {
      return fSections[0].fCoaxialCones->SurfaceAreaOfZPlanes();
    } else if (fSections.size() > 1) {
      CoaxialConesStruct<Precision> *firstSectionCones = fSections[0].fCoaxialCones;
      CoaxialConesStruct<Precision> *lastSectionCones  = fSections[fSections.size() - 1].fCoaxialCones;
      return (firstSectionCones->TotalSurfaceAreaOfLowerZPlanes() + lastSectionCones->TotalSurfaceAreaOfUpperZPlanes());
    }
    /*for(unsigned int i = 0 ; i < fSections.size() ; i++){
      CoaxialConesStruct<Precision> *CoaxialCones = fSections[i].fCoaxialCones;
    }*/
  }
#endif
  /*
    Precision Capacity(){
        Precision volume = 0.;
        for(unsigned int i = 0 ; i < fCoaxialConesStructVector.size() ; i++)
          volume += fCoaxialConesStructVector[i].Capacity();
        return volume;
      }
  */

  /*
    void Print(){
      std::cerr << "TotalNum Of Sections : " << fCoaxialConesStructVector.size() << std::endl;
      for(int i = 0 ; i < fCoaxialConesStructVector.size() ; i++){
        fCoaxialConesStructVector[i].Print();
      }
    }
  */
  VECCORE_ATT_HOST_DEVICE
  Precision Capacity()
  {
    Precision volume = 0.;
    for (unsigned int i = 0; i < fSections.size(); i++)
      volume += fSections[i].fCoaxialCones->Capacity();
    return volume;
  }
#if (0)
  Precision SurfaceArea()
  {
    // Precision surfArea = 0.;
    // TODO : Logic to calculate the Surface Area

    return (ConicalSurfaceArea() + SurfaceAreaOfZPlanes());
  }
#endif
  VECCORE_ATT_HOST_DEVICE
  int GetSectionIndex(Precision zposition) const
  {
    // TODO: consider binary search
    // TODO: consider making these comparisons tolerant in case we need it
    if (zposition < fZs[0]) return -1;
    for (unsigned int i = 0; i < fZs.size() - 1; ++i) {
      if (zposition >= fZs[i] && zposition <= fZs[i + 1]) return i;
    }
    return -2;
  }

  VECCORE_ATT_HOST_DEVICE
  int GetNSections() const { return fSections.size(); }

  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeSection const &GetSection(Precision zposition) const
  {
    // TODO: consider binary search
    int i = GetSectionIndex(zposition);
    return fSections[i];
  }

  VECCORE_ATT_HOST_DEVICE
  // GetSection if index is known
  GenericPolyconeSection const &GetSection(int index) const { return fSections[index]; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetZAtPlane(unsigned int index) const
  {
    assert(index <= fSections.size());
    return fZs[index];
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
