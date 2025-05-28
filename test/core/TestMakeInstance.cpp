#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/volumes/UnplacedCone.h"
#include "VecGeom/volumes/UnplacedOrb.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/GeoManager.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"
#include <cmath>

using namespace vecgeom;

// a unit test checking the factory mechanism to produce specialized unplaced
// volumes
int main()
{
  // BOX: IS TRIVIAL
  auto ubox = GeoManager::MakeInstance<UnplacedBox>(1., 1., 2.);
  VECGEOM_ASSERT(ubox != nullptr);
  VECGEOM_ASSERT(dynamic_cast<UnplacedBox *>(ubox));
  // let me try to make a specialized placed box
  Transformation3D placement(0, 0, 0);
  LogicalVolume lv("mybox", ubox);
  auto pv = lv.Place(&placement);
  VECGEOM_ASSERT(pv->Contains(Vector3D<double>(0, 0, 0)));

  // ORB: IS TRIVIAL
  auto uorb = GeoManager::MakeInstance<UnplacedOrb>(1.);
  VECGEOM_ASSERT(uorb != nullptr);
  VECGEOM_ASSERT(dynamic_cast<UnplacedOrb *>(uorb));

  // CHECK THE TUBE CASES
  {
    // an ordinary tube without inner radius
    auto utube = GeoManager::MakeInstance<UnplacedTube>(0., 1., 1., 0., 2. * M_PI);
    VECGEOM_ASSERT(utube != nullptr);
    VECGEOM_ASSERT(dynamic_cast<UnplacedTube *>(utube));
#ifndef VECGEOM_NO_SPECIALIZATION
    VECGEOM_ASSERT(dynamic_cast<SUnplacedTube<TubeTypes::NonHollowTube> *>(utube));
    VECGEOM_ASSERT(dynamic_cast<SUnplacedTube<TubeTypes::HollowTube> *>(utube) == nullptr);
#else
    VECGEOM_ASSERT(dynamic_cast<SUnplacedTube<TubeTypes::UniversalTube> *>(utube));
#endif

    // let me try to make a specialized placed hollow tube
    Transformation3D placement(0, 0, 0);
    LogicalVolume lv("mytube", utube);
    auto pv = lv.Place(&placement);
    auto c  = pv->Contains(Vector3D<double>(0, 0, 0));
    VECGEOM_ASSERT(c);
  }

  {
    // an ordinary hollow tube
    auto utube = GeoManager::MakeInstance<UnplacedTube>(0.5, 1., 1., 0., 2. * M_PI);
    VECGEOM_ASSERT(utube != nullptr);
    VECGEOM_ASSERT(dynamic_cast<UnplacedTube *>(utube));
#ifndef VECGEOM_NO_SPECIALIZATION
    VECGEOM_ASSERT(dynamic_cast<SUnplacedTube<TubeTypes::HollowTube> *>(utube));
    VECGEOM_ASSERT(dynamic_cast<SUnplacedTube<TubeTypes::NonHollowTube> *>(utube) == nullptr);
#else
    VECGEOM_ASSERT(dynamic_cast<SUnplacedTube<TubeTypes::UniversalTube> *>(utube));
#endif
  }

  // CHECK THE CONE CASES
  {
    // an ordinary cone without inner radii
    auto ucone = GeoManager::MakeInstance<UnplacedCone>(0., 1., 0., 1., 2., 0., kTwoPi);
    VECGEOM_ASSERT(ucone != nullptr);
    VECGEOM_ASSERT(dynamic_cast<UnplacedCone *>(ucone));
#ifndef VECGEOM_NO_SPECIALIZATION
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::NonHollowCone> *>(ucone));
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::HollowCone> *>(ucone) == nullptr);
#else
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::UniversalCone> *>(ucone));
#endif

    // let me try to make a specialized placed hollow cone
    Transformation3D placement(0, 0, 0);
    LogicalVolume lv("mycone", ucone);
    auto pv = lv.Place(&placement);
    auto c  = pv->Contains(Vector3D<double>(0, 0, 0));
    VECGEOM_ASSERT(c);
  }

  {
    // an ordinary hollow cone
    auto ucone = GeoManager::MakeInstance<UnplacedCone>(0.5, 1., 0.4, 1., 1.8, 0., kTwoPi);
    VECGEOM_ASSERT(ucone != nullptr);
    VECGEOM_ASSERT(dynamic_cast<UnplacedCone *>(ucone));
#ifndef VECGEOM_NO_SPECIALIZATION
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::HollowCone> *>(ucone));
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::NonHollowCone> *>(ucone) == nullptr);
#else
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::UniversalCone> *>(ucone));
#endif
  }

  {
    // a hollow cone with a smaller than PI sector
    auto ucone = GeoManager::MakeInstance<UnplacedCone>(0.5, 1., 0.4, 1., 1.8, 0., kPi / 3.);
    VECGEOM_ASSERT(ucone != nullptr);
    VECGEOM_ASSERT(dynamic_cast<UnplacedCone *>(ucone));
#ifndef VECGEOM_NO_SPECIALIZATION
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::NonHollowCone> *>(ucone) == nullptr);
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::HollowCone> *>(ucone) == nullptr);
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::HollowConeWithSmallerThanPiSector> *>(ucone));
#else
    VECGEOM_ASSERT(dynamic_cast<SUnplacedCone<ConeTypes::UniversalCone> *>(ucone));
#endif
  }

  std::cout << "test passed \n";
  return 0;
}
