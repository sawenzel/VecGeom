#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/volumes/kernel/shapetypes/TubeTypes.h"
#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/management/GeoManager.h"
//#include "ArgParser.h"

//.. ensure asserts are compiled in
#undef NDEBUG

using namespace vecgeom;

int main(int argc, char *argv[])
{
  /// OPTION_INT(npoints,1024);
  // OPTION_INT(nrep,1024);

  UnplacedBox worldUnplaced(10., 10., 10.);
  LogicalVolume world("world", &worldUnplaced);

  // components for boolean solid
  UnplacedBox motherbox(5., 5., 5.);
  GenericUnplacedTube subtractedtube(0.5, 2., 10., 0, kTwoPi);
  // translation for boolean solid right shape ( it should now stick outside )
  Transformation3D translation(-2.5, 0, 3.5);

  // VPlacedVolume *worldPlaced = world.Place();
  // GeoManager::Instance().SetWorld(worldPlaced);

  VPlacedVolume *placedsubtractedtube = (new LogicalVolume("", &subtractedtube))->Place(&translation);
  VPlacedVolume *placedmotherbox      = (new LogicalVolume("", &motherbox))->Place();

  // now make the unplaced boolean solid
  // UnplacedBooleanVolume booleansolid(kUnion, placedmotherbox, placedsubtractedtube);
  UnplacedBooleanVolume<kSubtraction> booleansolid(kSubtraction, placedmotherbox, placedsubtractedtube);
  LogicalVolume booleanlogical("booleanL", &booleansolid);

  // VPlacedVolume placedBooleanVolume = booleanlogical.Place();
  VPlacedVolume *placedBooleanVolume = (new LogicalVolume("", &booleansolid))->Place();
  VECGEOM_ASSERT(!placedBooleanVolume->GetUnplacedVolume()->IsConvex());

  // cleanup
  delete placedBooleanVolume;

  return 0;
}
