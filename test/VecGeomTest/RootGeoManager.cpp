// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \file RootGeoManager.cpp
/// \author created by Johannes de Fine Licht, Sandro Wenzel (CERN)

#include "RootGeoManager.h"

#include "PlacedRootVolume.h"
#include "UnplacedRootVolume.h"

#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/volumes/UnplacedCone.h"
#include "VecGeom/volumes/UnplacedParaboloid.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"
#include "VecGeom/volumes/UnplacedPolyhedron.h"
#include "VecGeom/volumes/UnplacedTrd.h"
#include "VecGeom/volumes/UnplacedOrb.h"
#include "VecGeom/volumes/UnplacedSphere.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/UnplacedTorus2.h"
#include "VecGeom/volumes/UnplacedTrapezoid.h"
#include "VecGeom/volumes/UnplacedPolycone.h"
#include "VecGeom/volumes/UnplacedScaledShape.h"
#include "VecGeom/volumes/UnplacedGenTrap.h"
#include "VecGeom/volumes/UnplacedHalfSpace.h"
#include "VecGeom/volumes/UnplacedSExtruVolume.h"
#include "VecGeom/volumes/UnplacedExtruded.h"
#include "VecGeom/volumes/PlanarPolygon.h"
#include "VecGeom/volumes/UnplacedAssembly.h"
#include "VecGeom/volumes/UnplacedCutTube.h"
#include "VecGeom/volumes/UnplacedTessellated.h"

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
#include "TGeoXtru.h"
#include "TGeoShapeAssembly.h"
#include "TGeoScaledShape.h"
#include "TGeoEltu.h"
#include "TGeoTessellated.h"
#include "TGeoHalfSpace.h"

#include <iostream>
#include <list>
#include <vector>

namespace vecgeom {

void RootGeoManager::LoadRootGeometry()
{
  Clear();
  GeoManager::Instance().Clear();
  TGeoNode const *const world_root = ::gGeoManager->GetTopNode();
  // Convert() will recursively convert daughters
  Stopwatch timer;
  timer.Start();
  fWorld = Convert(world_root);
  timer.Stop();
  if (fVerbose) {
    std::cout << "*** Conversion of ROOT -> VecGeom finished (" << timer.Elapsed() << " s) ***\n";
  }
  GeoManager::Instance().SetWorld(fWorld);
  timer.Start();
  GeoManager::Instance().CloseGeometry();
  timer.Stop();
  if (fVerbose) {
    std::cout << "*** Closing VecGeom geometry finished (" << timer.Elapsed() << " s) ***\n";
  }
  // fix the world --> close geometry might have changed it ( "compactification" )
  // this is very ugly of course: some observer pattern/ super smart pointer might be appropriate
  fWorld = GeoManager::Instance().GetWorld();

  // setup fast lookup table
  fTGeoNodeVector.resize(VPlacedVolume::GetIdCount(), nullptr);
  auto iter = fPlacedVolumeMap.begin();
  for (; iter != fPlacedVolumeMap.end(); ++iter) {
    fTGeoNodeVector[iter->first] = iter->second;
  }
}

void RootGeoManager::LoadRootGeometry(std::string filename)
{
  if (::gGeoManager != NULL) delete ::gGeoManager;
  TGeoManager::Import(filename.c_str());
  LoadRootGeometry();
}

bool RootGeoManager::ExportToROOTGeometry(VPlacedVolume const *topvolume, std::string filename)
{
  if (gGeoManager != nullptr && gGeoManager->IsClosed()) {
    std::cerr << "will not export to ROOT file; gGeoManager already initialized and closed\n";
    return false;
  }
  TGeoNode *world = Convert(topvolume);
  if (!world) {
    std::cerr << "The geometry cannot be converted to ROOT geometry\n";
    return false;
  }
  TGeoManager::SetVerboseLevel(0);
  ::gGeoManager->SetTopVolume(world->GetVolume());
  ::gGeoManager->CloseGeometry();
  //::gGeoManager->CheckOverlaps();
  ::gGeoManager->Export(filename.c_str());

  // setup fast lookup table
  fTGeoNodeVector.resize(VPlacedVolume::GetIdCount(), nullptr);
  auto iter = fPlacedVolumeMap.begin();
  for (; iter != fPlacedVolumeMap.end(); ++iter) {
    fTGeoNodeVector[iter->first] = iter->second;
  }
  return true;
}

// a helper function to convert ROOT assembly constructs into a flat list of nodes
// allows parsing of more complex ROOT geometries ( until VecGeom supports assemblies natively )
void FlattenAssemblies(TGeoNode *node, std::list<TGeoNode *> &nodeaccumulator, TGeoHMatrix const *globalmatrix,
                       int currentdepth, int &count /* keeps track of number of flattenened nodes */, int &maxdepth)
{
  maxdepth = currentdepth;
  if (RootGeoManager::Instance().GetFlattenAssemblies() &&
      nullptr != dynamic_cast<TGeoVolumeAssembly *>(node->GetVolume())) {
    // it is an assembly --> so modify the matrix
    TGeoVolumeAssembly *assembly = dynamic_cast<TGeoVolumeAssembly *>(node->GetVolume());
    for (int i = 0, nDaughters = assembly->GetNdaughters(); i < nDaughters; ++i) {
      TGeoHMatrix nextglobalmatrix = *globalmatrix;
      nextglobalmatrix.Multiply(assembly->GetNode(i)->GetMatrix());
      FlattenAssemblies(assembly->GetNode(i), nodeaccumulator, &nextglobalmatrix, currentdepth + 1, count, maxdepth);
    }
  } else {
    if (currentdepth == 0) // can keep original node ( it was not an assembly )
      nodeaccumulator.push_back(node);
    else { // need a new flattened node with a different transformation
      TGeoMatrix *newmatrix   = new TGeoHMatrix(*globalmatrix);
      TGeoNodeMatrix *newnode = new TGeoNodeMatrix(node->GetVolume(), newmatrix);
      newnode->SetNumber(node->GetNumber());

      // need a new name for the flattened node
      std::string *newname = new std::string(node->GetName());
      *newname += "_assemblyinternalcount_" + std::to_string(count);
      newnode->SetName(newname->c_str());
      nodeaccumulator.push_back(newnode);
      count++;
    }
  }
}

bool RootGeoManager::PostAdjustTransformation(Transformation3D *tr, TGeoNode const *node,
                                              Transformation3D *adjustment) const
{
  // post-fixing the placement ...
  // unfortunately, in ROOT there is a special case in which a placement transformation
  // is hidden (or "misused") inside the Box shape via the origin property
  //
  // --> In this case we need to adjust the transformation for this placement
  bool adjust(false);
  if (node->GetVolume()->GetShape()->IsA() == TGeoBBox::Class()) {
    TGeoBBox const *const box = static_cast<TGeoBBox const *>(node->GetVolume()->GetShape());
    auto o                    = box->GetOrigin();
    if (o[0] != 0. || o[1] != 0. || o[2] != 0.) {
      if (fVerbose) {
        std::cerr << "Warning: **********************************************************\n";
        std::cerr << "Warning: Found a box " << node->GetName() << " with non-zero origin\n";
        std::cerr << "Warning: **********************************************************\n";
      }
      *adjustment = Transformation3D::kIdentity;
      adjustment->SetTranslation(o[0] * LUnit(), o[1] * LUnit(), o[2] * LUnit());
      adjustment->SetProperties();
      tr->MultiplyFromRight(*adjustment);
      tr->SetProperties();
      adjust = true;
    }
  }

  // other special cases may follow ...
  return adjust;
}

VPlacedVolume *RootGeoManager::Convert(TGeoNode const *const node)
{
  if (fPlacedVolumeMap.Contains(node)) return const_cast<VPlacedVolume *>(GetPlacedVolume(node));
  // convert node transformation
  Transformation3D const *const transformation = Convert(node->GetMatrix());
  // possibly adjust transformation
  Transformation3D adjustmentTr;
  bool adjusted = PostAdjustTransformation(const_cast<Transformation3D *>(transformation), node, &adjustmentTr);

  LogicalVolume *const logical_volume = Convert(node->GetVolume());
  VPlacedVolume *placed_volume        = logical_volume->Place(node->GetName(), transformation);
  placed_volume->SetCopyNo(node->GetNumber());

  int remaining_daughters = 0;
  {
    // All or no daughters should have been placed already
    remaining_daughters = node->GetNdaughters() - logical_volume->GetDaughters().size();
    VECGEOM_ASSERT(remaining_daughters <= 0 || remaining_daughters == (int)node->GetNdaughters());
  }

  // we have to convert here assemblies to list of normal nodes
  std::list<TGeoNode *> flattenenednodelist;
  int assemblydepth   = 0;
  int flatteningcount = 0;
  for (int i = 0; i < remaining_daughters; ++i) {
    TGeoHMatrix trans = *node->GetDaughter(i)->GetMatrix();
    FlattenAssemblies(node->GetDaughter(i), flattenenednodelist, &trans, 0, flatteningcount, assemblydepth);
  }

  if ((int)flattenenednodelist.size() > remaining_daughters) {
    std::cerr << "INFO: flattening of assemblies (depth " << assemblydepth << ") resulted in "
              << flattenenednodelist.size() << " daughters vs " << remaining_daughters << "\n";
  }

  //
  for (auto &n : flattenenednodelist) {
    auto placed = Convert(n);
    logical_volume->PlaceDaughter(placed);

    // fixup placements in case the mother was shifted
    if (adjusted) {
      Transformation3D inv = adjustmentTr.Inverse();
      inv.SetProperties();
      Transformation3D *placedtr = const_cast<Transformation3D *>(placed->GetTransformation());
      inv.MultiplyFromRight(*placedtr);
      inv.SetProperties();
      placedtr->CopyFrom(inv);
    }
  }

  fPlacedVolumeMap.Set(node, placed_volume->id());
  return placed_volume;
}

TGeoNode *RootGeoManager::Convert(VPlacedVolume const *const placed_volume)
{
  if (fPlacedVolumeMap.Contains(placed_volume->id()))
    return const_cast<TGeoNode *>(fPlacedVolumeMap[placed_volume->id()]);

  TGeoVolume *geovolume = Convert(placed_volume, placed_volume->GetLogicalVolume());
  // Protect for volumes that may not be convertible in some cases (e.g. tessellated)
  if (!geovolume) return nullptr;
  TGeoNode *node = new TGeoNodeMatrix(geovolume, NULL);
  node->SetNumber(placed_volume->GetCopyNo());
  fPlacedVolumeMap.Set(node, placed_volume->id());

  // only need to do daughterloop once for every logical volume.
  // So only need to check if
  // logical volume already done ( if it already has the right number of daughters )
  auto remaining_daughters = placed_volume->GetDaughters().size() - geovolume->GetNdaughters();
  VECGEOM_ASSERT(remaining_daughters == 0 || remaining_daughters == placed_volume->GetDaughters().size());

  // do daughters
  for (size_t i = 0; i < remaining_daughters; ++i) {
    // get placed daughter
    VPlacedVolume const *daughter_placed = placed_volume->GetDaughters().operator[](i);

    // RECURSE DOWN HERE
    TGeoNode *daughternode = Convert(daughter_placed);
    if (!daughternode) return nullptr;

    // get matrix of daughter
    TGeoMatrix *geomatrixofdaughter = Convert(daughter_placed->GetTransformation());

    // add node to the TGeoVolume; using the TGEO API
    // unfortunately, there is not interface allowing to add an existing
    // nodepointer directly
    geovolume->AddNode(daughternode->GetVolume(), i, geomatrixofdaughter);
  }

  return node;
}

Transformation3D *RootGeoManager::Convert(TGeoMatrix const *const geomatrix)
{
  // if (fTransformationMap.Contains(geomatrix)) return const_cast<Transformation3D *>(fTransformationMap[geomatrix]);

  Double_t const *const t = geomatrix->GetTranslation();
  Double_t const *const r = geomatrix->GetRotationMatrix();

  Transformation3D *const transformation = new Transformation3D(t[0] * LUnit(), t[1] * LUnit(), t[2] * LUnit(), r[0],
                                                                r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);

  //  transformation->FixZeroes();
  // transformation->SetProperties();
  fTransformationMap.Set(geomatrix, transformation);
  return transformation;
}

TGeoMatrix *RootGeoManager::Convert(Transformation3D const *const trans)
{
  if (fTransformationMap.Contains(trans)) return const_cast<TGeoMatrix *>(fTransformationMap[trans]);

  TGeoMatrix *const geomatrix = Transformation3D::ConvertToTGeoMatrix(*trans);

  fTransformationMap.Set(geomatrix, trans);
  return geomatrix;
}

LogicalVolume *RootGeoManager::Convert(TGeoVolume const *const volume)
{
  if (fLogicalVolumeMap.Contains(volume)) return const_cast<LogicalVolume *>(fLogicalVolumeMap[volume]);

  VUnplacedVolume const *unplaced;
  if (!volume->IsAssembly()) {
    unplaced = Convert(volume->GetShape());
  } else {
    unplaced = ConvertAssembly(volume);
  }
  LogicalVolume *const logical_volume = new LogicalVolume(volume->GetName(), unplaced);
  // convert root material

  fLogicalVolumeMap.Set(volume, logical_volume);

  //// can be used to make a cross check for dimensions and other properties
  //// make a cross check using cubic volume property
  //  if (! (dynamic_cast<TGeoCompositeShape *>(volume->GetShape()) ||
  //		logical_volume->GetUnplacedVolume()->IsAssembly()))
  //  {
  //    const auto v1 = logical_volume->GetUnplacedVolume()->Capacity();
  //    const auto v2 = volume->Capacity();
  //    std::cerr << "v1 " << v1 << " " << v2 << "\n";
  //
  //    VECGEOM_ASSERT(v1 > 0.);
  //    VECGEOM_ASSERT(std::abs(v1 - v2 * LUnit() * LUnit() * LUnit())/v1 < 0.05);
  //  }
  return logical_volume;
}

// the inverse: here we need both the placed volume and logical volume as input
// they should match
TGeoVolume *RootGeoManager::Convert(VPlacedVolume const *const placed_volume, LogicalVolume const *const logical_volume)
{
  VECGEOM_ASSERT(placed_volume->GetLogicalVolume() == logical_volume);

  if (fLogicalVolumeMap.Contains(logical_volume)) return const_cast<TGeoVolume *>(fLogicalVolumeMap[logical_volume]);

  const TGeoShape *root_shape = placed_volume->ConvertToRoot();
  // Some shapes do not exist in ROOT: we need to protect for that
  if (!root_shape) return nullptr;
  TGeoVolume *geovolume = new TGeoVolume(logical_volume->GetLabel().c_str(), /* the name */
                                         root_shape, 0                       /* NO MATERIAL FOR THE MOMENT */
  );

  fLogicalVolumeMap.Set(geovolume, logical_volume);
  return geovolume;
}

VUnplacedVolume *RootGeoManager::Convert(TGeoShape const *const shape)
{

  if (fUnplacedVolumeMap.Contains(shape)) return const_cast<VUnplacedVolume *>(fUnplacedVolumeMap[shape]);

  VUnplacedVolume *unplaced_volume = NULL;

  // THE BOX
  if (shape->IsA() == TGeoBBox::Class()) {
    TGeoBBox const *const box = static_cast<TGeoBBox const *>(shape);

    unplaced_volume =
        GeoManager::MakeInstance<UnplacedBox>(box->GetDX() * LUnit(), box->GetDY() * LUnit(), box->GetDZ() * LUnit());
  }

  // THE HALF-SPACE
  if (shape->IsA() == TGeoHalfSpace::Class()) {
    TGeoHalfSpace *const halfspace = const_cast<TGeoHalfSpace *>(static_cast<TGeoHalfSpace const *>(shape));
    const double *point            = halfspace->GetPoint();
    const double *normal           = halfspace->GetNorm();
    unplaced_volume                = GeoManager::MakeInstance<UnplacedHalfSpace>(
        Vector3D<Precision>(point[0] * LUnit(), point[1] * LUnit(), point[2] * LUnit()),
        Vector3D<Precision>(normal[0], normal[1], normal[2]));
  }

  // THE TUBE
  if (shape->IsA() == TGeoTube::Class()) {
    TGeoTube const *const tube = static_cast<TGeoTube const *>(shape);

    unplaced_volume = GeoManager::MakeInstance<UnplacedTube>(tube->GetRmin() * LUnit(), tube->GetRmax() * LUnit(),
                                                             tube->GetDz() * LUnit(), 0., kTwoPi);
  }

  // THE TUBESEG
  if (shape->IsA() == TGeoTubeSeg::Class()) {
    TGeoTubeSeg const *const tube = static_cast<TGeoTubeSeg const *>(shape);

    unplaced_volume = GeoManager::MakeInstance<UnplacedTube>(tube->GetRmin() * LUnit(), tube->GetRmax() * LUnit(),
                                                             tube->GetDz() * LUnit(), kDegToRad * tube->GetPhi1(),
                                                             kDegToRad * (tube->GetPhi2() - tube->GetPhi1()));
  }

  // THE CONESEG
  if (shape->IsA() == TGeoConeSeg::Class()) {
    TGeoConeSeg const *const cone = static_cast<TGeoConeSeg const *>(shape);

    unplaced_volume = GeoManager::MakeInstance<UnplacedCone>(
        cone->GetRmin1() * LUnit(), cone->GetRmax1() * LUnit(), cone->GetRmin2() * LUnit(), cone->GetRmax2() * LUnit(),
        cone->GetDz() * LUnit(), kDegToRad * cone->GetPhi1(), kDegToRad * (cone->GetPhi2() - cone->GetPhi1()));
  }

  // THE CONE
  if (shape->IsA() == TGeoCone::Class()) {
    TGeoCone const *const cone = static_cast<TGeoCone const *>(shape);

    unplaced_volume = GeoManager::MakeInstance<UnplacedCone>(cone->GetRmin1() * LUnit(), cone->GetRmax1() * LUnit(),
                                                             cone->GetRmin2() * LUnit(), cone->GetRmax2() * LUnit(),
                                                             cone->GetDz() * LUnit(), 0., kTwoPi);
  }

  // THE PARABOLOID
  if (shape->IsA() == TGeoParaboloid::Class()) {
    TGeoParaboloid const *const p = static_cast<TGeoParaboloid const *>(shape);

    unplaced_volume = GeoManager::MakeInstance<UnplacedParaboloid>(p->GetRlo() * LUnit(), p->GetRhi() * LUnit(),
                                                                   p->GetDz() * LUnit());
  }

  // THE PARALLELEPIPED
  if (shape->IsA() == TGeoPara::Class()) {
    TGeoPara const *const p = static_cast<TGeoPara const *>(shape);

    unplaced_volume = GeoManager::MakeInstance<UnplacedParallelepiped>(
        p->GetX() * LUnit(), p->GetY() * LUnit(), p->GetZ() * LUnit(), kDegToRad * p->GetAlpha(),
        kDegToRad * p->GetTheta(), kDegToRad * p->GetPhi());
  }

  // Polyhedron/TGeoPgon
  if (shape->IsA() == TGeoPgon::Class()) {
    TGeoPgon const *pgon = static_cast<TGeoPgon const *>(shape);

    // fix dimensions - (requires making a copy of some arrays)
    const int NZs = pgon->GetNz();
    std::vector<Precision> zs(NZs);
    std::vector<Precision> rmins(NZs);
    std::vector<Precision> rmaxs(NZs);
    for (int i = 0; i < NZs; ++i) {
      zs[i]    = pgon->GetZ()[i] * LUnit();
      rmins[i] = pgon->GetRmin()[i] * LUnit();
      rmaxs[i] = pgon->GetRmax()[i] * LUnit();
    }

    unplaced_volume = GeoManager::MakeInstance<UnplacedPolyhedron>(pgon->GetPhi1() * kDegToRad, // phiStart
                                                                   pgon->GetDphi() * kDegToRad, // phiEnd
                                                                   pgon->GetNedges(),           // sideCount
                                                                   pgon->GetNz(),               // zPlaneCount
                                                                   zs.data(),                   // zPlanes
                                                                   rmins.data(),                // rMin
                                                                   rmaxs.data()                 // rMax
    );
  }

  // TRD2
  if (shape->IsA() == TGeoTrd2::Class()) {
    TGeoTrd2 const *const p = static_cast<TGeoTrd2 const *>(shape);

    unplaced_volume =
        GeoManager::MakeInstance<UnplacedTrd>(p->GetDx1() * LUnit(), p->GetDx2() * LUnit(), p->GetDy1() * LUnit(),
                                              p->GetDy2() * LUnit(), p->GetDz() * LUnit());
  }

  // TRD1
  if (shape->IsA() == TGeoTrd1::Class()) {
    TGeoTrd1 const *const p = static_cast<TGeoTrd1 const *>(shape);

    unplaced_volume = GeoManager::MakeInstance<UnplacedTrd>(p->GetDx1() * LUnit(), p->GetDx2() * LUnit(),
                                                            p->GetDy() * LUnit(), p->GetDz() * LUnit());
  }

  // TRAPEZOID
  if (shape->IsA() == TGeoTrap::Class()) {
    TGeoTrap const *const p = static_cast<TGeoTrap const *>(shape);
    if (!TGeoTrapIsDegenerate(p)) {
      unplaced_volume = GeoManager::MakeInstance<UnplacedTrapezoid>(
          p->GetDz() * LUnit(), p->GetTheta() * kDegToRad, p->GetPhi() * kDegToRad, p->GetH1() * LUnit(),
          p->GetBl1() * LUnit(), p->GetTl1() * LUnit(), p->GetAlpha1() * kDegToRad, p->GetH2() * LUnit(),
          p->GetBl2() * LUnit(), p->GetTl2() * LUnit(), p->GetAlpha2() * kDegToRad);
    } else {
      std::cerr << "Warning: this trap is degenerate -- will convert it to a generic trap!!\n";
      unplaced_volume = ToUnplacedGenTrap((TGeoArb8 const *)p);
    }
  }

  // THE SPHERE | ORB
  if (shape->IsA() == TGeoSphere::Class()) {
    // make distinction
    TGeoSphere const *const p = static_cast<TGeoSphere const *>(shape);
    if (p->GetRmin() == 0. && p->GetTheta2() - p->GetTheta1() == 180. && p->GetPhi2() - p->GetPhi1() == 360.) {
      unplaced_volume = GeoManager::MakeInstance<UnplacedOrb>(p->GetRmax() * LUnit());
    } else {
      unplaced_volume = GeoManager::MakeInstance<UnplacedSphere>(
          p->GetRmin() * LUnit(), p->GetRmax() * LUnit(), p->GetPhi1() * kDegToRad,
          (p->GetPhi2() - p->GetPhi1()) * kDegToRad, p->GetTheta1() * kDegToRad,
          (p->GetTheta2() - p->GetTheta1()) * kDegToRad);
    }
  }

  if (shape->IsA() == TGeoCompositeShape::Class()) {
    TGeoCompositeShape const *const compshape = static_cast<TGeoCompositeShape const *>(shape);
    TGeoBoolNode const *const boolnode        = compshape->GetBoolNode();

    // need the matrix;
    Transformation3D const *lefttrans  = Convert(boolnode->GetLeftMatrix());
    Transformation3D const *righttrans = Convert(boolnode->GetRightMatrix());

    auto transformationadjust = [this](Transformation3D *tr, TGeoShape const *shape) {
      // TODO: combine this with external method
      Transformation3D adjustment;
      if (shape->IsA() == TGeoBBox::Class()) {
        TGeoBBox const *const box = static_cast<TGeoBBox const *>(shape);
        auto o                    = box->GetOrigin();
        if (o[0] != 0. || o[1] != 0. || o[2] != 0.) {
          adjustment = Transformation3D::kIdentity;
          adjustment.SetTranslation(o[0] * LUnit(), o[1] * LUnit(), o[2] * LUnit());
          adjustment.SetProperties();
          tr->MultiplyFromRight(adjustment);
          tr->SetProperties();
        }
      }
    };
    // adjust transformation in case of shifted boxes
    transformationadjust(const_cast<Transformation3D *>(lefttrans), boolnode->GetLeftShape());
    transformationadjust(const_cast<Transformation3D *>(righttrans), boolnode->GetRightShape());

    // unplaced shapes
    VUnplacedVolume const *leftunplaced  = Convert(boolnode->GetLeftShape());
    VUnplacedVolume const *rightunplaced = Convert(boolnode->GetRightShape());

    VECGEOM_ASSERT(leftunplaced != nullptr);
    VECGEOM_ASSERT(rightunplaced != nullptr);

    // the problem is that I can only place logical volumes
    VPlacedVolume *const leftplaced  = (new LogicalVolume("inner_virtual", leftunplaced))->Place(lefttrans);
    VPlacedVolume *const rightplaced = (new LogicalVolume("inner_virtual", rightunplaced))->Place(righttrans);

    // now it depends on concrete type
    if (boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoSubtraction) {
      unplaced_volume =
          GeoManager::MakeInstance<UnplacedBooleanVolume<kSubtraction>>(kSubtraction, leftplaced, rightplaced);
    } else if (boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoIntersection) {
      unplaced_volume =
          GeoManager::MakeInstance<UnplacedBooleanVolume<kIntersection>>(kIntersection, leftplaced, rightplaced);
    } else if (boolnode->GetBooleanOperator() == TGeoBoolNode::kGeoUnion) {
      unplaced_volume = GeoManager::MakeInstance<UnplacedBooleanVolume<kUnion>>(kUnion, leftplaced, rightplaced);
    }
  }

  // THE TORUS
  if (shape->IsA() == TGeoTorus::Class()) {
    // make distinction
    TGeoTorus const *const p = static_cast<TGeoTorus const *>(shape);

    unplaced_volume =
        GeoManager::MakeInstance<UnplacedTorus2>(p->GetRmin() * LUnit(), p->GetRmax() * LUnit(), p->GetR() * LUnit(),
                                                 p->GetPhi1() * kDegToRad, p->GetDphi() * kDegToRad);
  }

  // THE POLYCONE
  if (shape->IsA() == TGeoPcon::Class()) {
    TGeoPcon const *const p = static_cast<TGeoPcon const *>(shape);

    // fix dimensions - (requires making a copy of some arrays)
    const int NZs = p->GetNz();
    std::vector<Precision> zs(NZs);
    std::vector<Precision> rmins(NZs);
    std::vector<Precision> rmaxs(NZs);
    for (int i = 0; i < NZs; ++i) {
      zs[i]    = p->GetZ()[i] * LUnit();
      rmins[i] = p->GetRmin()[i] * LUnit();
      rmaxs[i] = p->GetRmax()[i] * LUnit();
    }

    unplaced_volume = GeoManager::MakeInstance<UnplacedPolycone>(p->GetPhi1() * kDegToRad, p->GetDphi() * kDegToRad,
                                                                 p->GetNz(), zs.data(), rmins.data(), rmaxs.data());
  }

  // THE SCALED SHAPE
  if (shape->IsA() == TGeoScaledShape::Class()) {
    TGeoScaledShape const *const p = static_cast<TGeoScaledShape const *>(shape);
    // First convert the referenced shape
    VUnplacedVolume *referenced_shape = Convert(p->GetShape());
    const double *scale_root          = p->GetScale()->GetScale();

    unplaced_volume =
        GeoManager::MakeInstance<UnplacedScaledShape>(referenced_shape, scale_root[0], scale_root[1], scale_root[2]);
  }

  // THE ELLIPTICAL TUBE AS SCALED TUBE
  if (shape->IsA() == TGeoEltu::Class()) {
    TGeoEltu const *const p = static_cast<TGeoEltu const *>(shape);
    // Create the corresponding unplaced tube, with:
    // rmin=0, rmax=A, dz=dz, which is scaled with (1., A/B, 1.)
    UnplacedTube *tubeUnplaced =
        GeoManager::MakeInstance<UnplacedTube>(0, p->GetA() * LUnit(), p->GetDZ() * LUnit(), 0, kTwoPi);

    unplaced_volume = GeoManager::MakeInstance<UnplacedScaledShape>(tubeUnplaced, 1., p->GetB() / p->GetA(), 1.);
  }

  // THE ARB8
  if (shape->IsA() == TGeoArb8::Class() || shape->IsA() == TGeoGtra::Class()) {
    TGeoArb8 *p     = (TGeoArb8 *)(shape);
    unplaced_volume = ToUnplacedGenTrap(p);
  }

  // THE SIMPLE/GENERAL XTRU
  if (shape->IsA() == TGeoXtru::Class()) {
    TGeoXtru *p = (TGeoXtru *)(shape);
    // analyse convertability: the shape must have 2 planes with the same scaling
    // and offsets to make a simple extruded
    size_t Nvert = (size_t)p->GetNvert();
    size_t Nsect = (size_t)p->GetNz();
    if (Nsect == 2 && p->GetXOffset(0) == p->GetXOffset(1) && p->GetYOffset(0) == p->GetYOffset(1) &&
        p->GetScale(0) == p->GetScale(1)) {
      Precision *x = new Precision[Nvert];
      Precision *y = new Precision[Nvert];
      for (size_t i = 0; i < Nvert; ++i) {
        // Normally offsets should be 0 and scales should be 1, but just to be safe
        x[i] = LUnit() * (p->GetXOffset(0) + p->GetX(i) * p->GetScale(0));
        y[i] = LUnit() * (p->GetYOffset(0) + p->GetY(i) * p->GetScale(0));
      }
      unplaced_volume = GeoManager::MakeInstance<UnplacedSExtruVolume>(p->GetNvert(), x, y, LUnit() * p->GetZ(0),
                                                                       LUnit() * p->GetZ(1));
      delete[] x;
      delete[] y;
    }
#ifndef VECGEOM_CUDA_INTERFACE
    else {
      // Make the general extruded solid.
      XtruVertex2 *vertices = new XtruVertex2[Nvert];
      XtruSection *sections = new XtruSection[Nsect];
      for (size_t i = 0; i < Nvert; ++i) {
        vertices[i].x = p->GetX(i);
        vertices[i].y = p->GetY(i);
      }
      for (size_t i = 0; i < Nsect; ++i) {
        sections[i].fOrigin.Set(p->GetXOffset(i), p->GetYOffset(i), p->GetZ(i));
        sections[i].fScale = p->GetScale(i);
      }
      unplaced_volume = GeoManager::MakeInstance<UnplacedExtruded>(Nvert, vertices, Nsect, sections);
      delete[] vertices;
      delete[] sections;
    }
#endif
  }

  // THE CUT TUBE
  if (shape->IsA() == TGeoCtub::Class()) {
    TGeoCtub *ctube = (TGeoCtub *)(shape);
    // Create the corresponding unplaced cut tube
    // TODO: consider moving this as a specialization to UnplacedTube
    unplaced_volume = GeoManager::MakeInstance<UnplacedCutTube>(
        ctube->GetRmin(), ctube->GetRmax(), ctube->GetDz(), kDegToRad * ctube->GetPhi1(),
        kDegToRad * (ctube->GetPhi2() - ctube->GetPhi1()),
        Vector3D<Precision>(ctube->GetNlow()[0], ctube->GetNlow()[1], ctube->GetNlow()[2]),
        Vector3D<Precision>(ctube->GetNhigh()[0], ctube->GetNhigh()[1], ctube->GetNhigh()[2]));
  }

  // THE TESSELLATED
  if (shape->IsA() == TGeoTessellated::Class()) {
    TGeoTessellated *tes = (TGeoTessellated *)(shape);
    unplaced_volume      = GeoManager::MakeInstance<UnplacedTessellated>();
    auto unplaced_tes    = dynamic_cast<UnplacedTessellated *>(unplaced_volume);

    int nfacets{tes->GetNfacets()};
    for (int i = 0; i < nfacets; ++i) {
      auto &facet = tes->GetFacet(i);
      // for now only support the purely triangular case
      if (facet.GetNvert() != 3) {
        std::cerr << "Found unsupported facet in Tessellated conversion";
      }
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 32, 0)
      auto &v1 = tes->GetVertex(facet[0]);
      auto &v2 = tes->GetVertex(facet[1]);
      auto &v3 = tes->GetVertex(facet[2]);
#else
      auto &v1 = tes->GetVertex(facet.GetVertexIndex(0));
      auto &v2 = tes->GetVertex(facet.GetVertexIndex(1));
      auto &v3 = tes->GetVertex(facet.GetVertexIndex(2));
#endif
      Vector3D<Precision> v1_v(v1[0], v1[1], v1[2]);
      Vector3D<Precision> v2_v(v2[0], v2[1], v2[2]);
      Vector3D<Precision> v3_v(v3[0], v3[1], v3[2]);

      unplaced_tes->AddTriangularFacet(v1_v, v2_v, v3_v, true);
    }
    unplaced_tes->Close();
  }

  // New volumes should be implemented here...
  if (!unplaced_volume) {
    if (fVerbose) {
      printf("Unsupported shape for ROOT volume \"%s\" of type %s. "
             "Using ROOT implementation.\n",
             shape->GetName(), shape->ClassName());
    }
    unplaced_volume = new UnplacedRootVolume(shape);
  }

  fUnplacedVolumeMap.Set(shape, unplaced_volume);
  return unplaced_volume;
}

VUnplacedVolume *RootGeoManager::ConvertAssembly(TGeoVolume const *const v)
{
  if (TGeoVolumeAssembly const *va = dynamic_cast<TGeoVolumeAssembly const *>(v)) {
    // std::cerr << "treating volume assembly " << va->GetName() << "\n";
    (void)va;
    return new UnplacedAssembly();
  }
  return nullptr;
}

void RootGeoManager::PrintNodeTable() const
{
  for (auto iter : fPlacedVolumeMap) {
    std::cerr << iter.first << " " << iter.second << "\n";
    TGeoNode const *n = iter.second;
    n->Print();
  }
}

void RootGeoManager::Clear()
{
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
  for (auto i = fTransformationMap.begin(); i != fTransformationMap.end(); ++i) {
    delete i->first;
    delete i->second;
  }
  if (GeoManager::Instance().GetWorld() == fWorld) {
    GeoManager::Instance().SetWorld(nullptr);
  }
}

bool RootGeoManager::TGeoTrapIsDegenerate(TGeoTrap const *trap)
{
  bool degeneracy = false;
  // const_cast because ROOT is lacking a const GetVertices() function
  auto const vertices = const_cast<TGeoTrap *>(trap)->GetVertices();
  // check degeneracy within the layers (as vertices do not contains z information)
  for (int layer = 0; layer < 2; ++layer) {
    auto lowerindex = layer * 4;
    auto upperindex = (layer + 1) * 4;
    for (int i = lowerindex; i < upperindex; ++i) {
      auto currentx  = vertices[2 * i];
      auto current_y = vertices[2 * i + 1];
      for (int j = lowerindex; j < upperindex; ++j) {
        if (j == i) {
          continue;
        }
        auto otherx = vertices[2 * j];
        auto othery = vertices[2 * j + 1];
        if (otherx == currentx && othery == current_y) {
          degeneracy = true;
        }
      }
    }
  }
  return degeneracy;
}

UnplacedGenTrap *RootGeoManager::ToUnplacedGenTrap(TGeoArb8 const *p)
{
  // Create the corresponding GenTrap
  const double *vertices = const_cast<TGeoArb8 *>(p)->GetVertices();
  Precision verticesx[8], verticesy[8];
  for (auto ivert = 0; ivert < 8; ++ivert) {
    verticesx[ivert] = LUnit() * vertices[2 * ivert];
    verticesy[ivert] = LUnit() * vertices[2 * ivert + 1];
  }
  return GeoManager::MakeInstance<UnplacedGenTrap>(verticesx, verticesy, LUnit() * p->GetDz());
}

// lookup the placed volume corresponding to a TGeoNode
VPlacedVolume const *RootGeoManager::Lookup(TGeoNode const *node) const
{
  if (node == nullptr) return nullptr;
  return NavigationState::ToPlacedVolume(fPlacedVolumeMap[node]);
}

} // namespace vecgeom
