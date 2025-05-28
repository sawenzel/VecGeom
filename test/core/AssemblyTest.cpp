#include "VecGeom/volumes/UnplacedAssembly.h"
#include "VecGeom/volumes/PlacedAssembly.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/Box.h"
#include "test/unit_tests/ApproxEqual.h"

// make sure that assert are included even in Release mode
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"

using namespace vecgeom;

int main()
{

  // make a simple assembly of 2 boxes
  UnplacedBox *b    = new UnplacedBox(10., 10., 10.);
  LogicalVolume *lb = new LogicalVolume("boxlv", b);

  UnplacedAssembly *ass = new UnplacedAssembly();
  VECGEOM_ASSERT(ass->GetLogicalVolume() == nullptr);

  // this statement makes lv an assembly
  LogicalVolume *lv = new LogicalVolume("assemblylv", ass);
  VECGEOM_ASSERT(ass->GetLogicalVolume() == lv);

  // make simple assembly out of 2 placed boxes
  ass->AddVolume(lb->Place(new Transformation3D(-20., 0., 0.)));
  ass->AddVolume(lb->Place(new Transformation3D(20., 0., 0.)));
  VECGEOM_ASSERT(ass->GetNVolumes() == 2);

  // check that the bounding box is initialized
  VECGEOM_ASSERT(ass->GetLowerCorner().x() > -kInfLength);

  VPlacedVolume *const pv = lv->Place();

  // check the assembly property
  VECGEOM_ASSERT(pv->GetUnplacedVolume()->IsAssembly());
  VECGEOM_ASSERT(ass->IsAssembly());
  VECGEOM_ASSERT(!b->IsAssembly());

  GeoManager::Instance().SetWorld(pv);
  GeoManager::Instance().CloseGeometry();

  // verify correct conversion of Unplaced to Placed type
  PlacedAssembly const *pa = dynamic_cast<PlacedAssembly const *>(pv);
  VECGEOM_ASSERT(pa != nullptr);

  // some checks on Contains, Safety and DistanceToIn
  {
    Vector3D<Precision> p(0., 0., 0.);
    Vector3D<Precision> p2(20., 0., 0.);
    Vector3D<Precision> lp(0., 0., 0.);
    NavigationState *state = NavigationState::MakeInstance(10);
    // State must point to assembly parent (none in this case)
    state->Clear();
    std::cerr << pa->Contains(p, lp, *state) << "\n";
    state->Clear();
    VECGEOM_ASSERT(!pa->Contains(p, lp, *state));
    state->Clear();
    std::cerr << pa->Contains(p2, lp, *state) << "\n";
    state->Clear();
    VECGEOM_ASSERT(pa->Contains(p2, lp, *state));
    state->Clear();
    std::cerr << pa->Contains(p) << "\n";
    state->Clear();
    VECGEOM_ASSERT(!pa->Contains(p));
    state->Clear();
    std::cerr << pa->Contains(p2) << "\n";
    state->Clear();
    VECGEOM_ASSERT(pa->Contains(p2));

    VECGEOM_ASSERT(pa->SafetyToIn(Vector3D<Precision>(-10, 0, 0)) == 0.);
    VECGEOM_ASSERT(pa->SafetyToIn(Vector3D<Precision>(0, 0, 0)) == 10.);

    VECGEOM_ASSERT(pa->DistanceToIn(Vector3D<Precision>(-40, 0, 0), Vector3D<Precision>(1., 0, 0)) == 10);
    VECGEOM_ASSERT(pa->DistanceToIn(Vector3D<Precision>(0, -40, 0), Vector3D<Precision>(0, 1, 0)) == kInfLength);
    VECGEOM_ASSERT(pa->DistanceToIn(Vector3D<Precision>(0, 0, 0), Vector3D<Precision>(1., 0, 0)) == 10);
    VECGEOM_ASSERT(pa->DistanceToIn(Vector3D<Precision>(0, 0, 0), Vector3D<Precision>(-1., 0, 0)) == 10);
  }

  if (pv->GetUnplacedVolume()->IsAssembly()) {
    Vector3D<Precision> p(20., 0., 0.);
    Vector3D<Precision> lp(0., 0., 0.);
    NavigationState *state = NavigationState::MakeInstance(10);
    state->Clear();
    static_cast<PlacedAssembly const *>(pv)->Contains(p, lp, *state);
    state->Print();
  }

  // check Capacity and Surface Area
  VECGEOM_ASSERT(ass->Capacity() == 2. * b->Capacity());
  VECGEOM_ASSERT(ass->SurfaceArea() == 2. * b->SurfaceArea());
  VECGEOM_ASSERT(((PlacedAssembly *)pa)->Capacity() == 2. * b->Capacity());
  VECGEOM_ASSERT(((PlacedAssembly *)pa)->SurfaceArea() == 2. * b->SurfaceArea());

  // check Extent
  Vector3D<Precision> emin;
  Vector3D<Precision> emax;
  ass->Extent(emin, emax);
  VECGEOM_ASSERT(emin.x() <= -30);
  VECGEOM_ASSERT(emin.y() <= -10);
  VECGEOM_ASSERT(emin.z() <= -10);
  VECGEOM_ASSERT(emax.x() >= 30);
  VECGEOM_ASSERT(emax.y() >= 10);
  VECGEOM_ASSERT(emax.z() >= 10);

  VECGEOM_ASSERT(emin == ass->GetLowerCorner());
  VECGEOM_ASSERT(emax == ass->GetUpperCorner());

  // test bounding box
  Vector3D<Precision> minExtent, maxExtent;
  Vector3D<Precision> minBBox, maxBBox;
  ass->Extent(minExtent, maxExtent);
  ass->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual<Precision>(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual<Precision>(maxExtent, maxBBox));

  // check Normal
  // TBD

  // check some properties of BoxImplementation
  BoxStruct<Precision> bs(kInfLength, kInfLength, kInfLength);
  Vector3D<Precision> p(0., 0., 0.);
  Vector3D<Precision> d(1., 0., 0.);
  Precision dist;
  BoxImplementation::DistanceToIn(bs, p, d, kInfLength, dist);
  VECGEOM_ASSERT(dist < kInfLength);
  bool cont;
  BoxImplementation::Contains(bs, p, cont);
  VECGEOM_ASSERT(cont);
  std::cerr << "dist " << dist << "\n";

  Vector3D<Precision> corners[2];
  corners[0].Set(-kInfLength, -kInfLength, -kInfLength);
  corners[1].Set(kInfLength, kInfLength, kInfLength);
  VECGEOM_ASSERT(BoxImplementation::Intersect(corners, p, d, 0, kInfLength));

  return 0;
}
