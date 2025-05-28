#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/UnplacedBox.h"

// make sure that assert are included even in Release mode
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"
// Noddy assertion with message
#define assert_m(exp, msg) VECGEOM_ASSERT(((void)msg, exp))

using vecgeom::GeoManager;
using vecgeom::LogicalVolume;
using vecgeom::UnplacedBox;

// Create a basic geometry (well, just register create logical volumes)
int main()
{
  // Expect GeoManager to ...
  GeoManager& gm = GeoManager::Instance();

  // ... have no world
  assert_m(gm.GetWorld() == nullptr, "Default world must be nullptr");

  // ... have no logical volumes
  assert_m(gm.GetRegisteredVolumesCount() == 0, "Default geometry has zero logical volumes");


  // Expect that after creating a logical volume...
  UnplacedBox world_params(4., 4., 4.);
  LogicalVolume worldl("world", &world_params);

  // ... manager has one logical volume
  assert_m(gm.GetRegisteredVolumesCount() == 1, "Manager has 1 logical volumes after instantiating a LogicalVolume");

  // ... that is the one we expect
  assert_m(gm.FindLogicalVolume("world") == &worldl, "Manager must have the world logical volume");

  // ... with the correct id
  assert_m(gm.GetLogicalVolumeLabel(0) == "world", "First logical volume added must have id 0");

  // ... with round tripping giving the same answer
  assert_m(gm.GetLogicalVolumeId("world") == 0, "First logical volume added must have id 0");


  // Add two more
  UnplacedBox largebox_params(1.5, 1.5, 1.5);
  LogicalVolume largebox("Large box", &largebox_params);

  UnplacedBox smallbox_params(0.5, 0.5, 0.5);
  LogicalVolume smallbox("Small box", &smallbox_params);

  // Expect that ...
  // ... manager now has three lvs
  assert_m(gm.GetRegisteredVolumesCount() == 3, "Manager has 3 logical volumes after adding two more LogicalVolumes");

  // ... with expected id and label roundtrip
  assert_m(gm.GetLogicalVolumeLabel(1) == "Large box", "Second logical volume added must have id 1");
  assert_m(gm.GetLogicalVolumeId("Large box") == 1, "Second logical volume added must have id 1");

  assert_m(gm.GetLogicalVolumeLabel(2) == "Small box", "Third logical volume added must have id 2");
  assert_m(gm.GetLogicalVolumeId("Small box") == 2, "Third logical volume added must have id 2");


  // Expect that...
  // ... should get nullptr for unknown volume label
  assert_m(gm.FindLogicalVolume("foo") == nullptr, "Manager must return nullptr for unknown volumes");
  // ... should get nullptr out-of-range id
  assert_m(gm.FindLogicalVolume(42) == nullptr, "Manager must return nullptr for unknown volumes");
  // ... should get -1 for id of unknown volume, "" for name of out-of-range index
  assert_m(gm.GetLogicalVolumeId("foo") == -1, "Manager must return -1 as Id of unknown volume");
  assert_m(gm.GetLogicalVolumeLabel(42) == "", "Manager must return empty string as name of unknown volume");

  return 0;
}