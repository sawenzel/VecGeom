#include "../benchmark/ArgParser.h"
#include "VecGeomTest/ShapeTester.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Trapezoid.h"

using Precision     = vecgeom::Precision;
using Vec_t         = vecgeom::Vector3D<vecgeom::Precision>;
using VPlacedVolume = vecgeom::VPlacedVolume;
using VGTrap        = vecgeom::SimpleTrapezoid;

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat);

template <typename Trap_t>
Trap_t *buildFullTrap()
{
  return new Trap_t("FullTrap", 40, 0.12, 0.34, 15, 10, 10, 0, 30, 20, 20, 0);
}

template <typename Trap_t>
Trap_t *buildBoxLikeTrap(double dx, double dy, double dz)
{
  return new Trap_t("BoxLikeTrap", dz, 0, 0, dy, dx, dx, 0, dy, dx, dx, 0);
}

template <typename Trap_t>
Trap_t *buildTrapFromCorners()
{
  // validate construtor for input corner points -- add an xy-offset for non-zero theta,phi
  vecgeom::TrapCorners xyz;
  Precision xoffset = 9;
  Precision yoffset = -6;

  // define corner points
  // convention: p0(---); p1(+--); p2(-+-); p3(++-); p4(--+); p5(+-+); p6(-++); p7(+++)
  xyz[0] = Vec_t(-2 + xoffset, -5 + yoffset, -15);
  xyz[1] = Vec_t(2 + xoffset, -5 + yoffset, -15);
  xyz[2] = Vec_t(-3 + xoffset, 5 + yoffset, -15);
  xyz[3] = Vec_t(3 + xoffset, 5 + yoffset, -15);
  xyz[4] = Vec_t(-4 - xoffset, -10 - yoffset, 15);
  xyz[5] = Vec_t(4 - xoffset, -10 - yoffset, 15);
  xyz[6] = Vec_t(-6 - xoffset, 10 - yoffset, 15);
  xyz[7] = Vec_t(6 - xoffset, 10 - yoffset, 15);

  // create trapezoid
  return new Trap_t("slantedTrap", xyz);
}

template <typename Trap_t>
Trap_t *buildATrap(int type)
{

  switch (type) {
  case 0:
    std::cout << "Building default trapezoid\n";
    return buildFullTrap<Trap_t>();
    break;
  case 1:
    std::cout << "Building box-like trapezoid\n";
    return buildBoxLikeTrap<Trap_t>(10, 10, 10);
    break;
  case 2:
    std::cout << "Building trapezoid from corners\n";
    return buildTrapFromCorners<Trap_t>();
    break;
  default:
    std::cout << "*** No trap type provided.\n";
  }
  return 0;
}

int main(int argc, char *argv[])
{
  OPTION_INT(npoints, 10000);
  OPTION_BOOL(debug, false);
  OPTION_BOOL(stat, false);
  OPTION_INT(type, 2);

  auto trap = buildATrap<VGTrap>(type);
  trap->Print();
  return runTester<VPlacedVolume>(trap, npoints, debug, stat);
}

template <typename ImplT>
int runTester(ImplT const *shape, int npoints, bool debug, bool stat)
{
  ShapeTester<ImplT> tester;
  tester.setStat(stat);
  tester.SetMaxPoints(npoints);
  #ifdef VECGEOM_SINGLE_PRECISION
    tester.SetSolidTolerance(1e-4);
  #endif
  int errcode = tester.Run(shape);

  std::cout << "Final Error count for Shape *** " << shape->GetName() << "*** = " << errcode << "\n";
  std::cout << "=========================================================\n";

  if (shape) delete shape;
  return errcode;
}
