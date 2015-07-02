/// \file RootGeoManager.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "base/Transformation3D.h"
#include "management/GeoManager.h"
#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedRootVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/UnplacedBox.h"
#include "volumes/UnplacedTube.h"
#include "volumes/UnplacedCone.h"
#include "volumes/UnplacedRootVolume.h"
#include "volumes/UnplacedParaboloid.h"
#include "volumes/UnplacedParallelepiped.h"
#include "volumes/UnplacedPolyhedron.h"
#include "volumes/UnplacedTrd.h"
#include "volumes/UnplacedOrb.h"
#include "volumes/UnplacedSphere.h"
#include "volumes/UnplacedBooleanVolume.h"
//#include "volumes/UnplacedTorus.h"
#include "volumes/UnplacedTorus2.h"
#include "volumes/UnplacedTrapezoid.h"
#include "volumes/UnplacedPolycone.h"

#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoPara.h"
#include "TGeoParaboloid.h"
#include "TGeoPgon.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoTorus.h"
#include "TGeoArb8.h"
#include "TGeoPcon.h"

#include <cassert>

namespace vecgeom {

void RootGeoManager::LoadRootGeometry() {
  Clear();
  GeoManager::Instance().Clear();
  TGeoNode const *const world_root = ::gGeoManager->GetTopNode();
  // Convert() will recursively convert daughters
  fWorld = Convert(world_root);
  GeoManager::Instance().SetWorld(fWorld);
  GeoManager::Instance().CloseGeometry();
}


void RootGeoManager::LoadRootGeometry(std::string filename)
{
  if( ::gGeoManager != NULL ) delete ::gGeoManager;
  TGeoManager::Import(filename.c_str());
  LoadRootGeometry();
}

void RootGeoManager::ExportToROOTGeometry(VPlacedVolume const * topvolume,
        std::string filename )
{
   TGeoNode * world = Convert(topvolume);
   ::gGeoManager->SetTopVolume( world->GetVolume() );
   ::gGeoManager->CloseGeometry();
   ::gGeoManager->CheckOverlaps();
   ::gGeoManager->Export(filename.c_str());
}


VPlacedVolume* RootGeoManager::Convert(TGeoNode const *const node) {
  if (fPlacedVolumeMap.Contains(node))
      return const_cast<VPlacedVolume*> (fPlacedVolumeMap[node]);

  Transformation3D const *const transformation = Convert(node->GetMatrix());
  LogicalVolume *const logical_volume = Convert(node->GetVolume());
  VPlacedVolume *const placed_volume =
      logical_volume->Place(node->GetName(), transformation);

  int remaining_daughters = 0;
  {
    // All or no daughters should have been placed already
    remaining_daughters = node->GetNdaughters()
                          - logical_volume->GetDaughters().size();
    assert(remaining_daughters == 0 ||
           remaining_daughters == node->GetNdaughters());
  }
  for (int i = 0; i < remaining_daughters; ++i) {
    logical_volume->PlaceDaughter(Convert(node->GetDaughter(i)));
  }

  fPlacedVolumeMap.Set(node, placed_volume);
  return placed_volume;
}


TGeoNode* RootGeoManager::Convert(VPlacedVolume const *const placed_volume) {
  if (fPlacedVolumeMap.Contains(placed_volume))
      return const_cast<TGeoNode*> (fPlacedVolumeMap[placed_volume]);

  TGeoVolume * geovolume = Convert(placed_volume, placed_volume->GetLogicalVolume());
  TGeoNode * node = new TGeoNodeMatrix( geovolume, NULL );
  fPlacedVolumeMap.Set(node, placed_volume);

  // only need to do daughterloop once for every logical volume.
  // So only need to check if
  // logical volume already done ( if it already has the right number of daughters )
  int remaining_daughters = placed_volume->GetDaughters().size()
          - geovolume->GetNdaughters();
  assert(remaining_daughters == 0 ||
         remaining_daughters == placed_volume->GetDaughters().size());

  // do daughters
  for (int i = 0; i < remaining_daughters; ++i) {
      // get placed daughter
      VPlacedVolume const * daughter_placed =
              placed_volume->GetDaughters().operator[](i);

      // RECURSE DOWN HERE
      TGeoNode * daughternode = Convert( daughter_placed );

      // get matrix of daughter
      TGeoMatrix * geomatrixofdaughter
        = Convert(daughter_placed->GetTransformation());

      // add node to the TGeoVolume; using the TGEO API
      // unfortunately, there is not interface allowing to add an existing
      // nodepointer directly
      geovolume->AddNode( daughternode->GetVolume(), i, geomatrixofdaughter );
  }

  return node;
}


Transformation3D* RootGeoManager::Convert(TGeoMatrix const *const geomatrix) {
  if (fTransformationMap.Contains(geomatrix))
      return const_cast<Transformation3D*>(fTransformationMap[geomatrix]);

  Double_t const *const t = geomatrix->GetTranslation();
  Double_t const *const r = geomatrix->GetRotationMatrix();
  Transformation3D *const transformation =
      new Transformation3D(t[0], t[1], t[2], r[0], r[1], r[2],
                           r[3], r[4], r[5], r[6], r[7], r[8]);

  fTransformationMap.Set(geomatrix, transformation);
  return transformation;
}


TGeoMatrix* RootGeoManager::Convert(Transformation3D const *const trans) {
  if (fTransformationMap.Contains(trans))
      return const_cast<TGeoMatrix*>(fTransformationMap[trans]);

  TGeoMatrix *const geomatrix = trans->ConvertToTGeoMatrix();

  fTransformationMap.Set(geomatrix, trans);
  return geomatrix;
}



LogicalVolume* RootGeoManager::Convert(TGeoVolume const *const volume) {
  if (fLogicalVolumeMap.Contains(volume))
      return const_cast<LogicalVolume*>(fLogicalVolumeMap[volume]);

  VUnplacedVolume const *const unplaced = Convert(volume->GetShape());
  LogicalVolume *const logical_volume =
      new LogicalVolume(volume->GetName(), unplaced);

  fLogicalVolumeMap.Set(volume, logical_volume);
  return logical_volume;
}

// the inverse: here we need both the placed volume and logical volume as input
// they should match
TGeoVolume* RootGeoManager::Convert(VPlacedVolume const *const placed_volume,
        LogicalVolume const *const logical_volume) {
  assert( placed_volume->GetLogicalVolume() == logical_volume);

  if (fLogicalVolumeMap.Contains(logical_volume))
      return const_cast<TGeoVolume*>(fLogicalVolumeMap[logical_volume]);

  TGeoVolume * geovolume =
          new TGeoVolume(
            logical_volume->GetLabel().c_str(), /* the name */
            placed_volume->ConvertToRoot(),
            0 /* NO MATERIAL FOR THE MOMENT */
            );

  fLogicalVolumeMap.Set(geovolume, logical_volume);
  return geovolume;
}


VUnplacedVolume* RootGeoManager::Convert(TGeoShape const *const shape) {

  if (fUnplacedVolumeMap.Contains(shape))
      return const_cast<VUnplacedVolume*>(fUnplacedVolumeMap[shape]);

  VUnplacedVolume *unplaced_volume = NULL;

  // THE BOX
  if (shape->IsA() == TGeoBBox::Class()) {
    TGeoBBox const *const box = static_cast<TGeoBBox const*>(shape);
    unplaced_volume = new UnplacedBox(box->GetDX(), box->GetDY(), box->GetDZ());
  }

  // THE TUBE
  if (shape->IsA() == TGeoTube::Class()) {
    TGeoTube const *const tube = static_cast<TGeoTube const*>(shape);
    unplaced_volume = new UnplacedTube(tube->GetRmin(),
              tube->GetRmax(),
              tube->GetDz(),
              0.,kTwoPi);
  }

  // THE TUBESEG
  if (shape->IsA() == TGeoTubeSeg::Class()) {
      TGeoTubeSeg const *const tube = static_cast<TGeoTubeSeg const*>(shape);
      unplaced_volume = new UnplacedTube(tube->GetRmin(), tube->GetRmax(), tube->GetDz(),
              kDegToRad*tube->GetPhi1(),kDegToRad*(tube->GetPhi2()-tube->GetPhi1()));
  }


  // THE CONESEG
  if (shape->IsA() == TGeoConeSeg::Class()) {
      TGeoConeSeg const *const cone = static_cast<TGeoConeSeg const*>(shape);
      unplaced_volume = new UnplacedCone(cone->GetRmin1(),
              cone->GetRmax1(),
              cone->GetRmin2(),
              cone->GetRmax2(),
              cone->GetDz(),
              kDegToRad*cone->GetPhi1(),
              kDegToRad*(cone->GetPhi2()-cone->GetPhi1()));
  }

  // THE CONE
    if (shape->IsA() == TGeoCone::Class()) {
      TGeoCone const *const cone = static_cast<TGeoCone const*>(shape);
      unplaced_volume = new UnplacedCone(cone->GetRmin1(),
                cone->GetRmax1(),
                cone->GetRmin2(),
                cone->GetRmax2(),
                cone->GetDz(),
                0.,kTwoPi);
    }


  // THE PARABOLOID
  if (shape->IsA() == TGeoParaboloid::Class()) {
      TGeoParaboloid const *const p = static_cast<TGeoParaboloid const*>(shape);
      unplaced_volume = new UnplacedParaboloid(p->GetRlo(), p->GetRhi(), p->GetDz());
  }

  // THE PARALLELEPIPED
  if (shape->IsA() == TGeoPara::Class()) {
       TGeoPara const *const p = static_cast<TGeoPara const*>(shape);
       unplaced_volume = new UnplacedParallelepiped(p->GetX(), p->GetY(), p->GetZ(),
               p->GetAlpha(), p->GetTheta(), p->GetPhi());
  }

  // Polyhedron/TGeoPgon
  if (shape->IsA() == TGeoPgon::Class()) {
    TGeoPgon const *pgon = static_cast<TGeoPgon const*>(shape);
    unplaced_volume = new UnplacedPolyhedron(
      pgon->GetPhi1(),   // phiStart
      pgon->GetDphi(),   // phiEnd
      pgon->GetNedges(), // sideCount
      pgon->GetNz(),     // zPlaneCount
      pgon->GetZ(),      // zPlanes
      pgon->GetRmin(),   // rMin
      pgon->GetRmax()    // rMax
    );
  }

  // TRD2
  if (shape->IsA() == TGeoTrd2::Class() ) {
         TGeoTrd2 const *const p = static_cast<TGeoTrd2 const*>(shape);
         unplaced_volume = new UnplacedTrd(p->GetDx1(), p->GetDx2(), p->GetDy1(),
                 p->GetDy2(),p->GetDz());
  }

  // TRD1
  if (shape->IsA() == TGeoTrd1::Class() ) {
         TGeoTrd1 const *const p = static_cast<TGeoTrd1 const*>(shape);
         unplaced_volume = new UnplacedTrd(p->GetDx1(), p->GetDx2(), p->GetDy(),p->GetDz());
  }

  // TRAPEZOID
  if (shape->IsA() == TGeoTrap::Class() ) {
         TGeoTrap const *const p = static_cast<TGeoTrap const*>(shape);
         unplaced_volume = new UnplacedTrapezoid(p->GetDz(),p->GetTheta()*kDegToRad,p->GetPhi()*kDegToRad,
                                                 p->GetH1(),p->GetBl1(),p->GetTl1(),std::tan(p->GetAlpha1()*kDegToRad),
                                                 p->GetH2(),p->GetBl2(), p->GetTl2(),std::tan(p->GetAlpha2()*kDegToRad));
  }

  // THE SPHERE | ORB
  if (shape->IsA() == TGeoSphere::Class()) {
      // make distinction
      TGeoSphere const *const p = static_cast<TGeoSphere const*>(shape);
      if( p->GetRmin() == 0. &&
          p->GetTheta2() - p->GetTheta1() == 180. &&
          p->GetPhi2()   - p->GetPhi1()   == 360. ) {
          unplaced_volume = new UnplacedOrb(p->GetRmax());
      }
      else {
          unplaced_volume = new UnplacedSphere(p->GetRmin(),p->GetRmax(),
                  p->GetPhi1()*kDegToRad, (p->GetPhi2() - p->GetPhi1())*kDegToRad,
                  p->GetTheta1(), (p->GetTheta2()-p->GetTheta1())*kDegToRad);
      }
  }

  if (shape->IsA() == TGeoCompositeShape::Class()) {
    TGeoCompositeShape const *const compshape
         = static_cast<TGeoCompositeShape const*>(shape);
    TGeoBoolNode const *const boolnode = compshape->GetBoolNode();

     // need the matrix;
     Transformation3D const* lefttrans    = Convert( boolnode->GetLeftMatrix() );
     Transformation3D const* righttrans   = Convert( boolnode->GetRightMatrix() );
     // unplaced shapes
     VUnplacedVolume const* leftunplaced  = Convert( boolnode->GetLeftShape() );
     VUnplacedVolume const* rightunplaced = Convert( boolnode->GetRightShape() );

     // the problem is that I can only place logical volumes
     VPlacedVolume *const leftplaced =
          (new LogicalVolume("", leftunplaced ))->Place(lefttrans);

     VPlacedVolume *const rightplaced =
          (new LogicalVolume("", rightunplaced ))->Place(righttrans);

     // now it depends on concrete type
     if( boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoSubtraction ){
         unplaced_volume = new UnplacedBooleanVolume( kSubtraction,
             leftplaced, rightplaced);
     }
     else if( boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoIntersection ){
         unplaced_volume = new UnplacedBooleanVolume( kIntersection,
                      leftplaced, rightplaced);
     }
     else if( boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoUnion ){
         unplaced_volume = new UnplacedBooleanVolume( kUnion,
                      leftplaced, rightplaced);
     }
  }

  // THE TORUS
    if (shape->IsA() == TGeoTorus::Class()) {
        // make distinction
        TGeoTorus const *const p = static_cast<TGeoTorus const*>(shape);
            unplaced_volume = new UnplacedTorus2(p->GetRmin(),p->GetRmax(),
                    p->GetR(), p->GetPhi1()*kDegToRad, p->GetDphi()*kDegToRad);
    }


  // THE POLYCONE
   if (shape->IsA() == TGeoPcon::Class()) {
        TGeoPcon const *const p = static_cast<TGeoPcon const*>(shape);
        unplaced_volume = new UnplacedPolycone(
            p->GetPhi1()*kDegToRad,
            p->GetDphi()*kDegToRad,
            p->GetNz(),
            p->GetZ(),
            p->GetRmin(),
            p->GetRmax());
   }

   // New volumes should be implemented here...
  if (!unplaced_volume) {
    if (fVerbose) {
      printf("Unsupported shape for ROOT volume \"%s\". "
             "Using ROOT implementation.\n", shape->GetName());
    }
    unplaced_volume = new UnplacedRootVolume(shape);
  }

  fUnplacedVolumeMap.Set(shape, unplaced_volume);
  return unplaced_volume;
}


void RootGeoManager::PrintNodeTable() const
{
   for(auto iter : fPlacedVolumeMap)
   {
      std::cerr << iter.first << " " << iter.second << "\n";
      TGeoNode const * n = iter.second;
      n->Print();
   }
}

void RootGeoManager::Clear() {
  fPlacedVolumeMap.Clear();
  fUnplacedVolumeMap.Clear();
  fLogicalVolumeMap.Clear();
  fTransformationMap.Clear();
  // this should be done by smart pointers
//  for (auto i = fPlacedVolumeMap.begin(); i != fPlacedVolumeMap.end(); ++i) {
//    delete i->first;
//  }
//  for (auto i = fUnplacedVolumeMap.begin(); i != fUnplacedVolumeMap.end(); ++i) {
//    delete i->first;
//  }
//  for (auto i = fLogicalVolumeMap.begin(); i != fLogicalVolumeMap.end(); ++i) {
//    delete i->first;
//  }
//  for (auto i = fTransformationMap.begin(); i != fTransformationMap.end(); ++i) {
//    delete i->first;
//  }
  if (GeoManager::Instance().GetWorld() == fWorld) {
    GeoManager::Instance().SetWorld(nullptr);
  }
}

} // End global namespace
