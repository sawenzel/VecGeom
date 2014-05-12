#include "volumes/Parallelepiped.h"
#include "volumes/Paraboloid.h"
#include "volumes/Trapezoid.h"

using namespace vecgeom;

int main() {
  UnplacedParallelepiped unplacedPara
      = UnplacedParallelepiped(5, 5, 5, 5, 5, 5);
  LogicalVolume logicalPara = LogicalVolume(&unplacedPara);
  logicalPara.Place();
  UnplacedParaboloid unplacedLoid = UnplacedParaboloid();
  LogicalVolume logicalLoid = LogicalVolume(&unplacedLoid);
  logicalLoid.Place();
  UnplacedTrapezoid unplacedTrap = UnplacedTrapezoid();
  LogicalVolume logicalTrap = LogicalVolume(&unplacedTrap);
  logicalTrap.Place();
  return 0;
}