/// @file GenericPolyconeStruct.h
/// @brief Runtime storage for generic polycones.
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

/// @brief Storage for the generic polycone section decomposition.
/// @details The original `(r,z)` contour is split into ordered z sections.
/// Each section owns a `CoaxialConesStruct` shifted to its local z center.
/// The public implementation kernels use `fZs` to locate candidate sections
/// and then operate on the corresponding section helper.
template <typename T = double>
struct GenericPolyconeStruct {

  /// Per-section lower inner radii for the coaxial cone pieces.
  Vector<Vector<T>> fVectOfRmin1Vect;

  /// Per-section lower outer radii for the coaxial cone pieces.
  Vector<Vector<T>> fVectOfRmax1Vect;

  /// Per-section upper inner radii for the coaxial cone pieces.
  Vector<Vector<T>> fVectOfRmin2Vect;

  /// Per-section upper outer radii for the coaxial cone pieces.
  Vector<Vector<T>> fVectOfRmax2Vect;

  /// Per-section half-lengths in z.
  Vector<T> fVectOfDz;

  /// Start phi of the polycone.
  T fSPhi;

  /// Total phi opening of the polycone.
  T fDPhi;

  /// @brief Construct an empty generic polycone storage object.
  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeStruct() {}

  /// @brief Fill the section decomposition from reduced contour data.
  /// @details This internal setup function is called by the unplaced volume
  /// constructor after the input `(r,z)` contour has been reduced into coaxial
  /// cone pieces. It allocates one `CoaxialConesStruct` per z section, stores
  /// the section z shift, and caches z-plane positions and volume.
  /// @param vectOfRmin1Vect Per-section lower inner radii.
  /// @param vectOfRmax1Vect Per-section lower outer radii.
  /// @param vectOfRmin2Vect Per-section upper inner radii.
  /// @param vectOfRmax2Vect Per-section upper outer radii.
  /// @param zS Ordered z-plane coordinates.
  /// @param sPhi Start phi.
  /// @param dPhi Total phi opening.
  VECCORE_ATT_HOST_DEVICE
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

  /// Ordered z sections of the generic polycone.
  Vector<GenericPolyconeSection> fSections;

  /// Cached z-plane coordinates delimiting `fSections`.
  Vector<Precision> fZs;

  /// Cached surface area.
  T fSurfaceArea;

  /// Cached volume.
  T fCubicVolume;

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
  /// @brief Compute the volume from all section helpers.
  /// @return Sum of section capacities.
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
  /// @brief Locate the section containing a z coordinate.
  /// @details The return convention is `-1` for points below the first z plane
  /// and `-2` for points above the last z plane. Values on section boundaries
  /// are assigned to the first section whose interval contains the coordinate.
  /// @param zposition Polycone-local z coordinate.
  /// @return Section index, `-1` below the polycone, or `-2` above it.
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

  /// @brief Get the number of z sections.
  /// @return Number of section helpers.
  VECCORE_ATT_HOST_DEVICE
  int GetNSections() const { return fSections.size(); }

  /// @brief Get the section containing a z coordinate.
  /// @param zposition Polycone-local z coordinate.
  /// @return Section containing `zposition`.
  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeSection const &GetSection(Precision zposition) const
  {
    // TODO: consider binary search
    int i = GetSectionIndex(zposition);
    return fSections[i];
  }

  /// @brief Get a section by index.
  /// @param index Section index.
  /// @return Section at `index`.
  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeSection const &GetSection(int index) const { return fSections[index]; }

  /// @brief Get a cached z-plane coordinate.
  /// @param index Z-plane index in `[0, GetNSections()]`.
  /// @return Z coordinate of the requested plane.
  VECCORE_ATT_HOST_DEVICE
  Precision GetZAtPlane(unsigned int index) const
  {
    VECGEOM_ASSERT(index <= fSections.size());
    return fZs[index];
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif
