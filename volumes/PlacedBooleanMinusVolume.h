#ifndef VECGEOM_VOLUMES_PLACEDTBOOLEANMINUS_H_
#define VECGEOM_VOLUMES_PLACEDTBOOLEANMINUS_H_

#include "base/Global.h"
#include "backend/Backend.h"
 
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/kernel/BooleanMinusImplementation.h"
#include "volumes/UnplacedBooleanMinusVolume.h"
//#define VECGEOM_ROOT
//#define VECGEOM_BENCHMARK
#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoMatrix.h"
#endif
#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif

namespace VECGEOM_NAMESPACE {

class PlacedBooleanMinusVolume : public VPlacedVolume {

    typedef UnplacedBooleanMinusVolume UnplacedVol_t;

public:

#ifndef VECGEOM_NVCC
  PlacedBooleanMinusVolume(char const *const label,
            LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logicalVolume, transformation, boundingBox) {}

  PlacedBooleanMinusVolume(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : PlacedBooleanMinusVolume("", logicalVolume, transformation, boundingBox) {}
#else
  __device__
  PlacedBooleanMinusVolume(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox,
            const int id)
      : VPlacedVolume(logicalVolume, transformation, boundingBox, id) {}
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedBooleanMinusVolume() {}


  VECGEOM_CUDA_HEADER_BOTH
  UnplacedVol_t const* GetUnplacedVolume() const {
    return static_cast<UnplacedVol_t const *>(
        logical_volume()->unplaced_volume());
  }


  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const { };

  // CUDA specific
  virtual int memory_size() const { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation,
      VPlacedVolume *const gpu_ptr) const;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      Transformation3D const *const transformation) const;
#endif

  // Comparison specific

#ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const {
   return this;
  }

#ifdef VECGEOM_ROOT
  virtual TGeoShape const* ConvertToRoot() const {
      printf("Converting to ROOT\n");
      // what do we need?
      VPlacedVolume const * left = GetUnplacedVolume()->fLeftVolume;
      VPlacedVolume const * right = GetUnplacedVolume()->fRightVolume;
      Transformation3D const * leftm = left->transformation();
      Transformation3D const * rightm = right->transformation();
      TGeoRotation * leftRootRotation = new TGeoRotation();
      leftRootRotation->SetMatrix( leftm->Rotation() );
      TGeoRotation * rightRootRotation = new TGeoRotation();
          rightRootRotation->SetMatrix( rightm->Rotation() );
      TGeoCombiTrans * leftRootMatrix = new TGeoCombiTrans(
              leftm->Translation(0),
              leftm->Translation(1),
              leftm->Translation(2),
              leftRootRotation);
      TGeoCombiTrans * rightRootMatrix = new TGeoCombiTrans(
              rightm->Translation(0),
              rightm->Translation(1),
              rightm->Translation(2),
              rightRootRotation);

      // do some asserts that the transformations are correct

      TGeoSubtraction * node = new TGeoSubtraction(
              const_cast<TGeoShape*>(left->ConvertToRoot()),
              const_cast<TGeoShape*>(right->ConvertToRoot()),
              leftRootMatrix, rightRootMatrix);
      return new TGeoCompositeShape("RootComposite",node);
  }
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const {
      printf("Converting to USOLIDS\n");
      return new UBox("",10,10,10);
  }
#endif
#endif // VECGEOM_BENCHMARK

}; // end class declaration

} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDBOX_H_
