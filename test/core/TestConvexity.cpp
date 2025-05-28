/*Unit test to test the Convexity function for various shapes
 *
 */
#undef NDEBUG
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Sphere.h"
//#include "ApproxEqual.h"

#include "VecGeom/volumes/Orb.h"
#include "VecGeom/volumes/Sphere.h"
#include "VecGeom/volumes/Paraboloid.h"
#include "VecGeom/volumes/Cone.h"
#include "VecGeom/volumes/Torus2.h"
#include "VecGeom/volumes/Tube.h"
#include "VecGeom/volumes/Parallelepiped.h"
#include "VecGeom/volumes/Trd.h"
#include "VecGeom/volumes/Polycone.h"
#include "VecGeom/volumes/Trapezoid.h"
#include "VecGeom/volumes/Polyhedron.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/ScaledShape.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/volumes/Hype.h"

#include <cmath>
#include <iomanip>

#define PI 3.14159265358979323846
#define deg PI / 180.

using namespace VECGEOM_NAMESPACE;

bool test_ConvexityOrb()
{

  vecgeom::SimpleOrb b1("Solid VecGeomOrb #1", 5.);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());
  return true;
}

bool test_ConvexitySphere()
{

  Precision rmin = 0., rmax = 5., sphi = 0., dphi = 2 * PI, stheta = 0., dtheta = PI;
  vecgeom::SimpleSphere b1("Solide VecGeomSphere #1", rmin, rmax, sphi, dphi, stheta, dtheta);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b2("Solide VecGeomSphere #2", rmin, rmax, 0., PI, stheta, dtheta);
  VECGEOM_ASSERT(b2.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b3("Solide VecGeomSphere #3", rmin, rmax, 0., PI / 3, stheta, dtheta);
  VECGEOM_ASSERT(b3.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b4("Solide VecGeomSphere #4", rmin, rmax, 0., 4 * PI / 3, stheta, dtheta);
  VECGEOM_ASSERT(!b4.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b5("Solide VecGeomSphere #5", rmin, rmax, PI / 3., PI / 3, stheta, dtheta);
  VECGEOM_ASSERT(b5.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b6("Solide VecGeomSphere #6", rmin, rmax, PI / 3., 2 * PI / 3, stheta, dtheta);
  VECGEOM_ASSERT(b6.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b7("Solide VecGeomSphere #7", rmin, rmax, PI / 3., PI, stheta, dtheta);
  VECGEOM_ASSERT(b7.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b8("Solide VecGeomSphere #8", rmin, rmax, PI / 3., 7 * PI / 6, stheta, dtheta);
  VECGEOM_ASSERT(!b8.GetUnplacedVolume()->IsConvex());

  // THESE TESTS ARE TESTING WEDGE BEHAVIOUR --> MOVE THEM INTO A WEDGE TEST

  /*
  //checking proper dphi calculation if specified dphi>2PI
  //Should be accepted by Wedge
  //Convention used for dPhi is if(dPhi>2PI) dPhi=2PI //needs a relook
  vecgeom::SimpleSphere b9("Solide VecGeomSphere #9", rmin, rmax, PI/3. , 4*PI, stheta, dtheta);
  VECGEOM_ASSERT(b9.GetUnplacedVolume()->IsConvex());
  //std::cerr<<"Newly Calcuated DPHi of b9 : "<<b9.GetDPhi()<<std::endl;

  vecgeom::SimpleSphere b10("Solide VecGeomSphere #10", rmin, rmax, PI/3. , 5*PI, stheta, dtheta);
  VECGEOM_ASSERT(b10.GetUnplacedVolume()->IsConvex());
  //std::cerr<<"Newly Calcuated DPHi of b10 : "<<b10.GetDPhi()<<std::endl;


  //This case should be discussed
  vecgeom::SimpleSphere b11("Solide VecGeomSphere #11", rmin, rmax, PI/3. , ((2*PI) + (7*PI/6)), stheta, dtheta);
  VECGEOM_ASSERT(b11.GetUnplacedVolume()->IsConvex());
  //std::cerr<<"Newly Calcuated DPHi of b11 : "<<b10.GetDPhi()<<std::endl;
  */

  vecgeom::SimpleSphere b12("Solide VecGeomSphere #12", rmin, rmax, 0., 2 * PI, stheta, PI / 2);
  VECGEOM_ASSERT(b12.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b13("Solide VecGeomSphere #13", rmin, rmax, 0., 2 * PI, stheta, 2 * PI / 3);
  VECGEOM_ASSERT(!b13.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b14("Solide VecGeomSphere #14", rmin, rmax, 0., 2 * PI, stheta, PI / 3);
  VECGEOM_ASSERT(b14.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b15("Solide VecGeomSphere #15", rmin, rmax, 0., 2 * PI, PI / 6, PI / 6);
  VECGEOM_ASSERT(!b15.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b16("Solide VecGeomSphere #16", rmin, rmax, 0., 2 * PI, PI / 6, PI / 3);
  VECGEOM_ASSERT(!b16.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b17("Solide VecGeomSphere #17", rmin, rmax, 0., 2 * PI, PI / 2, PI / 2);
  VECGEOM_ASSERT(b17.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b18("Solide VecGeomSphere #18", rmin, rmax, 0., 2 * PI, PI / 2, PI / 6);
  VECGEOM_ASSERT(!b18.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b19("Solide VecGeomSphere #19", rmin, rmax, 0., 2 * PI, 2 * PI / 3, PI / 3);
  VECGEOM_ASSERT(b19.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b20("Solide VecGeomSphere #20", rmin, rmax, 0., 2 * PI, 2 * PI / 3, PI / 6);
  VECGEOM_ASSERT(!b20.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleSphere b21("Solide VecGeomSphere #21", rmin, rmax, 0., 2 * PI, PI / 3, 2 * PI / 3);
  VECGEOM_ASSERT(!b21.GetUnplacedVolume()->IsConvex());

  vecgeom::SimpleSphere b22("Solide VecGeomSphere #22", rmin, rmax, 0., 2 * PI, PI / 3, PI / 3);
  VECGEOM_ASSERT(!b22.GetUnplacedVolume()->IsConvex());

  vecgeom::SimpleSphere b23("Solide VecGeomSphere #23", 3, rmax, 0., 2 * PI, 0, PI);
  VECGEOM_ASSERT(!b23.GetUnplacedVolume()->IsConvex());

  vecgeom::SimpleSphere b24("Solide VecGeomSphere #24", 3, rmax, 0., 2 * PI / 3, PI / 3, PI / 3);
  VECGEOM_ASSERT(!b24.GetUnplacedVolume()->IsConvex());

  return true;
}

bool test_ConvexityParaboloid()
{

  vecgeom::SimpleParaboloid b1("VecGeomParaboloid", 5., 8., 10.);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleParaboloid b2("VecGeomParaboloid", 0., 8., 10.);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());
  return true;
}

bool test_ConvexityCone()
{

  Precision rmin1 = 0., rmax1 = 5., rmin2 = 0., rmax2 = 7., dz = 10., sphi = 0., dphi = 2 * PI;
  vecgeom::SimpleCone b1("VecGeomCone1", rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleCone b2("VecGeomCone2", 2., rmax1, rmin2, rmax2, dz, sphi, dphi);
  VECGEOM_ASSERT(!b2.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleCone b3("VecGeomCone3", rmin1, rmax1, 4., rmax2, dz, sphi, dphi);
  VECGEOM_ASSERT(!b3.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleCone b4("VecGeomCone4", 2., rmax1, 4., rmax2, dz, sphi, dphi);
  VECGEOM_ASSERT(!b4.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleCone b5("VecGeomCone5", rmin1, rmax1, rmin2, rmax2, dz, sphi, PI);
  VECGEOM_ASSERT(b5.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleCone b6("VecGeomCone6", rmin1, rmax1, rmin2, rmax2, dz, sphi, PI / 3);
  VECGEOM_ASSERT(b6.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleCone b7("VecGeomCone7", rmin1, rmax1, rmin2, rmax2, dz, sphi, 4 * PI / 3);
  VECGEOM_ASSERT(!b7.GetUnplacedVolume()->IsConvex());

  return true;
}

bool test_ConvexityTorus()
{

  Precision rmin = 0., rmax = 5., rtor = 0., sphi = 0., dphi = 2 * PI;
  vecgeom::SimpleTorus2 b1("VecGeomTorus1", rmin, rmax, rtor, sphi, dphi); // Torus becomes Orb
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTorus2 b2("VecGeomTorus2", 3, rmax, rtor, sphi, dphi); // Torus becomes SphericalShell
  VECGEOM_ASSERT(!b2.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTorus2 b3("VecGeomTorus3", 3, rmax, 15, sphi, dphi); // Real Complete Torus
  VECGEOM_ASSERT(!b3.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTorus2 b4("VecGeomTorus4", rmin, rmax, rtor, sphi, PI);
  VECGEOM_ASSERT(b4.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTorus2 b5("VecGeomTorus5", rmin, rmax, rtor, sphi, PI / 3);
  VECGEOM_ASSERT(b5.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTorus2 b6("VecGeomTorus6", rmin, rmax, rtor, sphi, 4 * PI / 3);
  VECGEOM_ASSERT(!b6.GetUnplacedVolume()->IsConvex());

  return true;
}

bool test_ConvexityTube()
{

  Precision rmin = 0., rmax = 5., dz = 10., sphi = 0., dphi = 2 * PI;
  vecgeom::SimpleTube b1("VecgeomTube1", rmin, rmax, dz, sphi, dphi); // Solid Cylinder
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTube b2("VecgeomTube2", 3, rmax, dz, sphi, dphi); // Hollow Cylinder
  VECGEOM_ASSERT(!b2.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTube b3("VecgeomTube3", rmin, rmax, dz, sphi, PI);
  VECGEOM_ASSERT(b3.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTube b4("VecgeomTube4", rmin, rmax, dz, sphi, 2 * PI / 3);
  VECGEOM_ASSERT(b4.GetUnplacedVolume()->IsConvex());
  vecgeom::SimpleTube b5("VecgeomTube5", rmin, rmax, dz, sphi, 4 * PI / 3);
  VECGEOM_ASSERT(!b5.GetUnplacedVolume()->IsConvex());

  return true;
}

bool test_ConvexityParallelepiped()
{
  Precision dx = 20., dy = 30., dz = 40., alpha = 30., theta = 15., phi = 30.;
  vecgeom::SimpleParallelepiped b1("VecGeomParallelepiped1", dx, dy, dz, alpha, theta, phi);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());

  return true;
}

bool test_ConvexityTrd()
{
  Precision xlower = 20., xupper = 10., ylower = 15., yupper = 15, dz = 40.;
  vecgeom::SimpleTrd b1("VecGeomParallelepiped1", xlower, xupper, ylower, yupper, dz);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());

  return true;
}

bool test_ConvexityPolycone()
{

  Precision phiStart = 0., deltaPhi = kTwoPi / 3;
  int nZ = 10;
  // Precision rmin[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  Precision rmin[10] = {6., 0., 0., 0., 15., 0., 0., 0., 3., 15.};
  Precision rmax[10] = {10., 10., 10., 20., 20., 10., 10., 5., 5., 20.};
  Precision z[10]    = {-20., 0., 0., 20., 20., 40., 45., 50., 50., 60.};

  // Precision rmin[4]={0.,0.,0.,0.};
  // Precision rmax[4]={15.,15.,15.,10.};
  // Precision z[4]={0.,20.,30.,40.};

  vecgeom::SimplePolycone b1("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
  VECGEOM_ASSERT(!b1.GetUnplacedVolume()->IsConvex());

  // Added this new test case provided by Guilherme lima and Phillipe,
  int nZ2            = 5;
  Precision rmin2[5] = {0., 0., 0., 0., 0.};
  // Precision rmax[5] = {5.,10.,10.,20.,20.};   // bad
  Precision rmax2[5] = {15., 20., 20., 10., 5.}; // bad
  Precision z2[5]    = {0., 10., 20., 20., 40.};
  phiStart           = 0.;
  deltaPhi           = kTwoPi;
  vecgeom::SimplePolycone b2("VecGeomPolycone2", phiStart, deltaPhi, nZ2, z2, rmin2, rmax2);
  VECGEOM_ASSERT(!b2.GetUnplacedVolume()->IsConvex());

  int nZ3            = 6;
  Precision rmin3[6] = {0., 0., 0., 0., 0., 0.};
  Precision rmax3[6] = {10., 20., 20., 30., 20., 10.};
  Precision z3[6]    = {0., 10., 20., 20., 20., 30.};
  vecgeom::SimplePolycone b3("VecGeomPolycone3", phiStart, deltaPhi, nZ3, z3, rmin3, rmax3);
  VECGEOM_ASSERT(b3.GetUnplacedVolume()->IsConvex());

  int nZ4            = 3;
  Precision rmin4[3] = {0., 0., 0.};
  Precision rmax4[3] = {10., 20., 30.};
  Precision z4[3]    = {0., 10., 10.};
  vecgeom::SimplePolycone b4("VecGeomPolycone3", phiStart, deltaPhi, nZ4, z4, rmin4, rmax4);
  VECGEOM_ASSERT(b4.GetUnplacedVolume()->IsConvex());

  // Some more test cases
  // Convex polycone,
  {
    // constexpr int nZ = 0;
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 6;
      Precision rmin[nZ] = {0., 0., 0., 0., 0., 0.};
      Precision rmax[nZ] = {10., 20., 20., 30., 20., 10.};
      Precision z[nZ]    = {0., 10., 20., 20., 20., 30.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(B.GetUnplacedVolume()->IsConvex());
    }

    // The below two test cases are very important because of introduction
    // of non zero inner radius for some sections of polycone.

    // Not Convex
    // Because of non zero inner radius at some section, which
    // exist in the final polycone also
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 6;
      Precision rmin[nZ] = {0., 0., 10., 0., 0., 0.};
      Precision rmax[nZ] = {10., 20., 20., 30., 20., 10.};
      Precision z[nZ]    = {0., 10., 20., 20., 20., 30.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(!B.GetUnplacedVolume()->IsConvex());
    }

    // Convex
    // Even if a non zero inner radius at some section,
    // but this inner radius does not exist in the final polycone.
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 6;
      Precision rmin[nZ] = {0., 0., 0., 10., 0., 0.};
      Precision rmax[nZ] = {10., 20., 20., 30., 20., 10.};
      Precision z[nZ]    = {0., 10., 20., 20., 20., 30.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(B.GetUnplacedVolume()->IsConvex());
    }

    // Not Convex
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 8;
      Precision rmin[nZ] = {0., 0., 0., 0., 0., 0., 0., 0.};
      Precision rmax[nZ] = {10., 20., 20., 10., 10., 5., 5., 20.};
      Precision z[nZ]    = {0., 20., 20., 40., 45., 50., 50., 60.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(!B.GetUnplacedVolume()->IsConvex());
    }

    // Not Convex
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 6;
      Precision rmin[nZ] = {0., 0., 0., 0., 0., 0.};
      Precision rmax[nZ] = {10., 20., 30., 25., 20., 30.};
      Precision z[nZ]    = {0., 10., 10., 10., 20., 30.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(!B.GetUnplacedVolume()->IsConvex());
    }

    // Simple, important and nice test case of non convex polycone.
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 3;
      Precision rmin[nZ] = {0., 0., 0.};
      Precision rmax[nZ] = {10., 20., 30.};
      Precision z[nZ]    = {0., 10., 15.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(!B.GetUnplacedVolume()->IsConvex());
    }

    // Same as above but this time nature is convex because of increased Z with same Rmax
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 3;
      Precision rmin[nZ] = {0., 0., 0.};
      Precision rmax[nZ] = {10., 20., 30.};
      Precision z[nZ]    = {0., 10., 30.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(B.GetUnplacedVolume()->IsConvex());
    }

    // Not Convex
    // Introducing non-zero rmin at the starting section
    {
      deltaPhi           = kTwoPi;
      constexpr int nZ   = 3;
      Precision rmin[nZ] = {5., 0., 0.};
      Precision rmax[nZ] = {10., 20., 30.};
      Precision z[nZ]    = {0., 10., 30.};
      vecgeom::SimplePolycone B("VecGeomPolycone", phiStart, deltaPhi, nZ, z, rmin, rmax);
      VECGEOM_ASSERT(!B.GetUnplacedVolume()->IsConvex());
    }
  }

  return true;
}

bool test_ConvexityTrapezoid()
{

  vecgeom::SimpleTrapezoid b1("trap3", 50, 0, 0, 50, 50, 50, PI / 4, 50, 50, 50, PI / 4);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());

  return true;
}

bool test_ConvexityPolyhedron()
{

  Precision phiStart = 0., deltaPhi = 170 * kDegToRad;
  int sides                  = 4; //, nZ=10;
  constexpr int nPlanes      = 4;
  Precision zPlanes[nPlanes] = {-2, -1, 1, 2};
  Precision rInner[nPlanes]  = {0, 0, 0, 0};
  Precision rOuter[nPlanes]  = {2, 2, 2, 2};

  vecgeom::SimplePolyhedron b1("Vecgeom Polyhedron", phiStart, deltaPhi, sides, nPlanes, zPlanes, rInner, rOuter);
  VECGEOM_ASSERT(b1.GetUnplacedVolume()->IsConvex());

  vecgeom::SimplePolyhedron b2("Vecgeom Polyhedron", phiStart, 60 * kDegToRad, sides, nPlanes, zPlanes, rInner, rOuter);
  VECGEOM_ASSERT(b2.GetUnplacedVolume()->IsConvex());

  vecgeom::SimplePolyhedron b3("Vecgeom Polyhedron", phiStart, 200 * kDegToRad, sides, nPlanes, zPlanes, rInner,
                               rOuter);
  VECGEOM_ASSERT(!b3.GetUnplacedVolume()->IsConvex());

  rOuter[1] = 1.;
  rOuter[2] = 1.;
  vecgeom::SimplePolyhedron b5("Vecgeom Polyhedron", phiStart, 120 * kDegToRad, sides, nPlanes, zPlanes, rInner,
                               rOuter);
  VECGEOM_ASSERT(!b5.GetUnplacedVolume()->IsConvex());

  rOuter[0] = 1.;
  rOuter[1] = 2.;
  rOuter[2] = 2.;
  rOuter[3] = 1.;
  vecgeom::SimplePolyhedron b6("Vecgeom Polyhedron", phiStart, 120 * kDegToRad, sides, nPlanes, zPlanes, rInner,
                               rOuter);
  VECGEOM_ASSERT(b6.GetUnplacedVolume()->IsConvex());

  rInner[1] = 1.;
  rInner[2] = 0.5;
  vecgeom::SimplePolyhedron b4("Vecgeom Polyhedron", phiStart, 60 * kDegToRad, sides, nPlanes, zPlanes, rInner, rOuter);
  VECGEOM_ASSERT(!b4.GetUnplacedVolume()->IsConvex());

  return true;
}
bool test_ConvexityHype()
{
  vecgeom::SimpleHype b1("Solid VecGeomHype #1", 10., 15., PI / 4, PI / 3, 50);
  VECGEOM_ASSERT(!b1.GetUnplacedVolume()->IsConvex());

  vecgeom::SimpleHype b2("Solid VecGeomHype #2", 0., 15., 0., PI / 3, 50);
  VECGEOM_ASSERT(!b2.GetUnplacedVolume()->IsConvex());

  // Case when hype becomes Solid Tube
  vecgeom::SimpleHype b3("Solid VecGeomHype #3", 0., 15., 0., 0., 50);
  VECGEOM_ASSERT(b3.GetUnplacedVolume()->IsConvex());

  return true;
}
//_________________________________________________________________________________________
// Convexity test for scaled Shapes

bool test_ConvexityScaledOrb()
{
  vecgeom::SimpleOrb orb("Visualizer Orb", 3);
  vecgeom::SimpleScaledShape scaledOrb("Scaled Orb", orb.GetUnplacedVolume(), 0.5, 1.2, 1.);
  VECGEOM_ASSERT(scaledOrb.GetUnplacedVolume()->IsConvex());

  return true;
}

//_________________________________________________________________________________________

int main()
{

  VECGEOM_ASSERT(test_ConvexityOrb());
  VECGEOM_ASSERT(test_ConvexitySphere());
  VECGEOM_ASSERT(test_ConvexityParaboloid());
  VECGEOM_ASSERT(test_ConvexityCone());
  VECGEOM_ASSERT(test_ConvexityTorus());
  VECGEOM_ASSERT(test_ConvexityTube());
  VECGEOM_ASSERT(test_ConvexityParallelepiped());
  VECGEOM_ASSERT(test_ConvexityTrd());
  // VECGEOM_ASSERT(test_Convexity_Y());
  VECGEOM_ASSERT(test_ConvexityPolycone());
  VECGEOM_ASSERT(test_ConvexityTrapezoid());
  VECGEOM_ASSERT(test_ConvexityPolyhedron());
  VECGEOM_ASSERT(test_ConvexityHype());

  // Test for ScaledShapes
  test_ConvexityScaledOrb();

  std::cout << "------------------------------" << std::endl;
  std::cout << "--- Convexity Tests Passed ---" << std::endl;
  std::cout << "------------------------------" << std::endl;

  return 0;
}
