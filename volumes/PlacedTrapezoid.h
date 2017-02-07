/*
 * @file   volumes/PlacedTrapezoid.h
 * @author Guilherme Lima (lima 'at' fnal 'dot' gov)
 *
 * 2014-05-01 - Created, based on the Parallelepiped draft
 */

#ifndef VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_

#include "base/Global.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedTrapezoid.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedTrapezoid; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedTrapezoid );

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedTrapezoid : public VPlacedVolume {

public:

  typedef UnplacedTrapezoid UnplacedShape_t;

#ifndef VECGEOM_NVCC

  PlacedTrapezoid(char const *const label,
                  LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logical_volume, transformation, boundingBox) {}

  PlacedTrapezoid(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox)
    : PlacedTrapezoid("", logical_volume, transformation, boundingBox) {}

#else

  __device__
  PlacedTrapezoid(LogicalVolume const *const logical_volume,
                  Transformation3D const *const transformation,
                  PlacedBox const *const boundingBox, const int id)
      : VPlacedVolume(logical_volume, transformation, boundingBox, id) {}

#endif
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedTrapezoid();

  /// Accessors
  /// @{

  /* Retrieves the unplaced volume pointer from the logical volume and casts it
   * to an unplaced trapezoid.
   */
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid const* GetUnplacedVolume() const {
    return static_cast<UnplacedTrapezoid const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDz() const { return GetUnplacedVolume()->GetDz(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTheta() const { return GetUnplacedVolume()->GetTheta(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetPhi() const { return GetUnplacedVolume()->GetPhi(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDy1() const { return GetUnplacedVolume()->GetDy1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx1() const { return GetUnplacedVolume()->GetDx1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx2() const { return GetUnplacedVolume()->GetDx2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha1() const { return GetUnplacedVolume()->GetTanAlpha1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDy2() const { return GetUnplacedVolume()->GetDy2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx3() const { return GetUnplacedVolume()->GetDx3(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDx4() const { return GetUnplacedVolume()->GetDx4(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetTanAlpha2() const { return GetUnplacedVolume()->GetTanAlpha2(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha1() const { return GetUnplacedVolume()->GetAlpha1(); }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetAlpha2() const { return GetUnplacedVolume()->GetAlpha2(); }

#ifndef VECGEOM_NVCC
  virtual
  Precision Capacity() override { return GetUnplacedVolume()->Capacity(); }

  virtual
  Precision SurfaceArea() override { return GetUnplacedVolume()->SurfaceArea();}

  bool Normal(Vector3D<Precision> const & point, Vector3D<Precision> & normal ) const override {
    return GetUnplacedVolume()->Normal(point, normal);
  }

  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

  Vector3D<Precision>  GetPointOnSurface() const override {
    return GetUnplacedVolume()->GetPointOnSurface();
  }

#if defined(VECGEOM_USOLIDS)
  std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType() ;}
#endif
#endif

  VECGEOM_CUDA_HEADER_BOTH
  void ComputeBoundingBox();

  VECGEOM_CUDA_HEADER_BOTH
  void GetParameterList() const { return GetUnplacedVolume()->GetParameterList() ;}

#if defined(VECGEOM_USOLIDS)
  VECGEOM_CUDA_HEADER_BOTH
  std::ostream& StreamInfo(std::ostream &os) const override {
    return GetUnplacedVolume()->StreamInfo(os);
  }
#endif

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const override;
#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const override;
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const override;
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const override;
#endif
#endif // VECGEOM_NVCC

protected:

  // static PlacedBox* make_bounding_box(LogicalVolume const *const logical_volume,
  //                                     Transformation3D const *const transformation) {

  //   UnplacedTrapezoid const *const utrap = static_cast<UnplacedTrapezoid const *const>(logical_volume->unplaced_volume());
  //   UnplacedBox const *const unplaced_bbox = new UnplacedBox(
  //     std::max(std::max(utrap->GetDx1(),utrap->GetDx2()),std::max(utrap->GetDx3(),utrap->GetDx4())),
  //     std::max(utrap->GetDy1(),utrap->GetDy2()), utrap->GetDz());
  //   LogicalVolume const *const box_volume =  new LogicalVolume(unplaced_bbox);
  //   return new PlacedBox(box_volume, transformation);
  // }

}; // end of class PlacedTrapezoid

} } // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTRAPEZOID_H_
