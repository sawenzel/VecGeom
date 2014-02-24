#include "volumes/logical_volume.h"
#include "volumes/box.h"

using namespace vecgeom;

int main() {

  UnplacedBox world_params = UnplacedBox(4., 4., 4.);
  UnplacedBox largebox_params = UnplacedBox(1.5, 1.5, 1.5);
  UnplacedBox smallbox_params = UnplacedBox(0.5, 0.5, 0.5);

  LogicalVolume world = LogicalVolume(world_params);
  LogicalVolume largebox = LogicalVolume(largebox_params);
  LogicalVolume smallbox = LogicalVolume(smallbox_params);

  TransformationMatrix origin = TransformationMatrix();
  TransformationMatrix box1 = TransformationMatrix( 2,  2,  2);
  TransformationMatrix box2 = TransformationMatrix(-2,  2,  2);
  TransformationMatrix box3 = TransformationMatrix( 2, -2,  2);
  TransformationMatrix box4 = TransformationMatrix( 2,  2, -2);
  TransformationMatrix box5 = TransformationMatrix(-2, -2,  2);
  TransformationMatrix box6 = TransformationMatrix(-2,  2, -2);
  TransformationMatrix box7 = TransformationMatrix( 2, -2, -2);
  TransformationMatrix box8 = TransformationMatrix(-2, -2, -2);

  largebox.PlaceDaughter(smallbox, origin);
  world.PlaceDaughter(largebox, box1);
  world.PlaceDaughter(largebox, box2);
  world.PlaceDaughter(largebox, box3);
  world.PlaceDaughter(largebox, box4);
  world.PlaceDaughter(largebox, box5);
  world.PlaceDaughter(largebox, box6);
  world.PlaceDaughter(largebox, box7);
  world.PlaceDaughter(largebox, box8);

  world.PrintContent();

  return 0;
}