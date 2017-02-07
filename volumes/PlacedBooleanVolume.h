#ifndef VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_
#define VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_

#include "base/Global.h"
#include "backend/Backend.h"
 
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedBooleanVolume.h"
#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#endif
#ifdef VECGEOM_GEANT4
#include "G4SubtractionSolid.hh"
#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#endif


namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class PlacedBooleanVolume; )
VECGEOM_DEVICE_DECLARE_CONV( PlacedBooleanVolume );

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedBooleanVolume : public VPlacedVolume {

    typedef UnplacedBooleanVolume UnplacedVol_t;

public:

#ifndef VECGEOM_NVCC
  PlacedBooleanVolume(char const *const label,
            LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : VPlacedVolume(label, logicalVolume, transformation, boundingBox) {}

  PlacedBooleanVolume(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox)
      : PlacedBooleanVolume("", logicalVolume, transformation, boundingBox) {}
#else
  __device__
  PlacedBooleanVolume(LogicalVolume const *const logicalVolume,
            Transformation3D const *const transformation,
            PlacedBox const *const boundingBox,
            const int id)
      : VPlacedVolume(logicalVolume, transformation, boundingBox, id) {}
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~PlacedBooleanVolume() {}


  VECGEOM_CUDA_HEADER_BOTH
  UnplacedVol_t const* GetUnplacedVolume() const {
    return static_cast<UnplacedVol_t const *>(
        GetLogicalVolume()->GetUnplacedVolume());
  }

//#ifndef VECGEOM_NVCC
  virtual Precision Capacity() override {
       // TODO: implement this
      return 0.;
  }

  void Extent(Vector3D<Precision>& aMin, Vector3D<Precision>& aMax) const override {
    GetUnplacedVolume()->Extent(aMin, aMax);
  }

#if defined(VECGEOM_USOLIDS)
  std::string GetEntityType() const override { return GetUnplacedVolume()->GetEntityType() ;}
#endif

  virtual Vector3D<Precision> GetPointOnSurface() const override;
//#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const override { };

  // CUDA specific
  virtual int memory_size() const override { return sizeof(*this); }

  // Comparison specific

#ifndef VECGEOM_NVCC
  virtual VPlacedVolume const* ConvertToUnspecialized() const override {
   return this;
  }
#ifdef VECGEOM_ROOT
 virtual TGeoShape const* ConvertToRoot() const override {
      // printf("Converting to ROOT\n");
      // what do we need?
      VPlacedVolume const * left = GetUnplacedVolume()->fLeftVolume;
      VPlacedVolume const * right = GetUnplacedVolume()->fRightVolume;
      Transformation3D const * leftm = left->GetTransformation();
      Transformation3D const * rightm = right->GetTransformation();

      TGeoShape *shape = NULL;
      if( GetUnplacedVolume()->GetOp() == kSubtraction ){
        TGeoSubtraction * node = new TGeoSubtraction(
              const_cast<TGeoShape*>(left->ConvertToRoot()),
              const_cast<TGeoShape*>(right->ConvertToRoot()),
              leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
        shape = new TGeoCompositeShape("RootComposite",node);
      }
      if( GetUnplacedVolume()->GetOp() == kUnion ){
        TGeoUnion * node = new TGeoUnion(
              const_cast<TGeoShape*>(left->ConvertToRoot()),
              const_cast<TGeoShape*>(right->ConvertToRoot()),
              leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
        shape = new TGeoCompositeShape("RootComposite",node);
      }
      if( GetUnplacedVolume()->GetOp() == kIntersection ){
        TGeoIntersection * node = new TGeoIntersection(
              const_cast<TGeoShape*>(left->ConvertToRoot()),
              const_cast<TGeoShape*>(right->ConvertToRoot()),
              leftm->ConvertToTGeoMatrix(), rightm->ConvertToTGeoMatrix());
        shape = new TGeoCompositeShape("RootComposite",node);
      }
      return shape;
  }
#endif
#ifdef VECGEOM_USOLIDS
  virtual ::VUSolid const* ConvertToUSolids() const override {
    // currently not supported in USOLIDS -- returning NULL
      return nullptr;
  }
#endif
#ifdef VECGEOM_GEANT4
  virtual G4VSolid const* ConvertToGeant4() const override {
      VPlacedVolume const * left = GetUnplacedVolume()->fLeftVolume;
      VPlacedVolume const * right = GetUnplacedVolume()->fRightVolume;
      Transformation3D const * rightm = right->GetTransformation();
      G4RotationMatrix * g4rot = new G4RotationMatrix();
      g4rot->set( CLHEP::HepRep3x3( rightm->Rotation() ) );
      if( GetUnplacedVolume()->GetOp() == kSubtraction ){
            return new G4SubtractionSolid( GetLabel(),
                 const_cast<G4VSolid*>(left->ConvertToGeant4()),
                 const_cast<G4VSolid*>(right->ConvertToGeant4()),
                 g4rot,
                 G4ThreeVector(rightm->Translation(0),  rightm->Translation(1),  rightm->Translation(2)));
      }
      if( GetUnplacedVolume()->GetOp() == kUnion ){
             return new G4UnionSolid( GetLabel(),
                 const_cast<G4VSolid*>(left->ConvertToGeant4()),
                 const_cast<G4VSolid*>(right->ConvertToGeant4()),
                 g4rot,
                 G4ThreeVector(rightm->Translation(0),  rightm->Translation(1),  rightm->Translation(2)));
      }
      if( GetUnplacedVolume()->GetOp() == kIntersection ){
              return new G4IntersectionSolid( GetLabel(),
                 const_cast<G4VSolid*>(left->ConvertToGeant4()),
                 const_cast<G4VSolid*>(right->ConvertToGeant4()),
                 g4rot,
                 G4ThreeVector(rightm->Translation(0),  rightm->Translation(1),  rightm->Translation(2)));
       }
      return NULL;
  }
  #endif
#endif // VECGEOM_BENCHMARK

}; // end class declaration

} // End impl namespace
} // End global namespace

#endif // VECGEOM_VOLUMES_PLACEDTBOOLEAN_H_
