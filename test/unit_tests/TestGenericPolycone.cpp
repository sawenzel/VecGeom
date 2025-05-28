/// @file TestGenericPolycone.cpp
/// @author Raman Sehgal (raman.sehgal@cern.ch)

// ensure asserts are compiled in
#undef NDEBUG

#include <iomanip>
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
//#include "VecGeom/volumes/EllipticUtilities.h"
#include "VecGeom/volumes/CoaxialCones.h"
#include "VecGeom/volumes/GenericPolycone.h"
#include "ApproxEqual.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/volumes/GenericPolyconeStruct.h"
#include "VecGeom/base/FpeEnable.h"
bool testvecgeom = false;

using vecgeom::kInfLength;
using vecgeom::kTolerance;
using vecgeom::Precision;

template <class GenericPolycone_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestGenericPolycone()
{
  /*
   * Add the required unit test
   *
   */

  /* Creating a test Simple GenericPolycone by rotating the following contour
   *       ------------------------
   *       |                      |
   *       | /------------------\ |
   *       |/                    \|
   */

  bool verbose       = true;
  Precision startPhi = 0., deltaPhi = 2. * M_PI; /// 3.;
  constexpr int numRz = 6;
  Precision r[numRz]  = {1., 2., 4., 5., 5., 1.};
  Precision z[numRz]  = {1., 2., 2., 1., 3., 3.};
  GenericPolycone_t Simple("GenericPolycone", startPhi, deltaPhi, numRz, r, z);
  Vec_t aMin, aMax;
  Simple.GetUnplacedVolume()->Extent(aMin, aMax);
  std::cout << "Extent : " << aMin << " : " << aMax << std::endl;

  std::cout << "Capacity : " << Simple.GetUnplacedVolume()->Capacity() << std::endl;
  std::cout << "SurfaceArea : " << Simple.GetUnplacedVolume()->SurfaceArea() << std::endl;

  VECGEOM_ASSERT(Simple.Inside(Vec_t(2., 0., 2.5)) == vecgeom::EInside::kInside);
  // std::cout << "Location of Inside point (2.,0.,2.5) using Contains : " << Simple.Contains(Vec_t(2., 0., 2.5))
  //        << std::endl;
  VECGEOM_ASSERT(Simple.Contains(Vec_t(2., 0., 2.5)));

  VECGEOM_ASSERT(Simple.Inside(Vec_t(1.5, 0., 1.9999)) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(Simple.Inside(Vec_t(4.5, 0., 1.9999)) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(Simple.Inside(Vec_t(2.1, 0., 1.9999)) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(Simple.Inside(Vec_t(5.1, 0., 1.9999)) == vecgeom::EInside::kOutside);

  // std::cout << "Location using Contains : " << Simple.Contains(Vec_t(5.1, 0., 1.9999)) << std::endl;
  VECGEOM_ASSERT(!Simple.Contains(Vec_t(5.1, 0., 1.9999)));
  VECGEOM_ASSERT(Simple.Inside(Vec_t(2., 0., 3.)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(Vec_t(5., 0., 2.1)) == vecgeom::EInside::kSurface);

  std::cout << "Surface Point (1.,0.,2.) : Location of point which is on edge of both Cones Sections : "
            << Simple.Inside(Vec_t(1., 0., 2.)) << std::endl;
  VECGEOM_ASSERT(Simple.Inside(Vec_t(1., 0., 2.)) == vecgeom::EInside::kSurface);
  std::cout << "Surface Point (5.,0.,2.) : Location of point which is on edge of both Cones Sections : "
            << Simple.Inside(Vec_t(5., 0., 2.)) << std::endl;
  VECGEOM_ASSERT(Simple.Inside(Vec_t(5., 0., 2.)) == vecgeom::EInside::kSurface);

  // Corner Point
  VECGEOM_ASSERT(Simple.Inside(Vec_t(1., 0., 1.)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(Vec_t(5., 0., 1.)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(Vec_t(1., 0., 3.)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(Simple.Inside(Vec_t(5., 0., 3.)) == vecgeom::EInside::kSurface);

  // DistanceToIn tests
  VECGEOM_ASSERT(Simple.DistanceToIn(Vec_t(3., 0., 1.5), Vec_t(0., 0., 1.)) == 0.5);

  Vec_t outPt1(8., 0., 1.5);
  Vec_t dir(-1., 0., 0.);
  Precision Dist = Simple.DistanceToIn(outPt1, dir);
  // std::cout << std::setprecision(20) << "DistanceToIn of (8.,0.,1.5) : " << Dist << std::endl;
  VECGEOM_ASSERT(Dist == 3);
  VECGEOM_ASSERT(Simple.Inside(outPt1 + dir * Dist) == vecgeom::EInside::kSurface);

  Vec_t outPt2(3., 0., 0.);
  VECGEOM_ASSERT(Simple.Inside(outPt2) == vecgeom::EInside::kOutside);
  Vec_t dir2(Vec_t(4., 0., 2) - outPt2);
  // dir /= dir.Mag();//.Normalize();
  dir2.Normalize();
  Dist = Simple.DistanceToIn(outPt2, dir2);
  if (verbose) {
    std::cout << "Distance : " << Dist << std::endl;
    std::cout << "Moved Point : " << (outPt2 + dir2 * Dist) << std::endl;
    std::cout << "Location of Moved point which is actually a corner point and again having a common edge of both "
                 "Cones Sections  : "
              << Simple.Inside(outPt2 + dir2 * Dist) << std::endl;
  }
  VECGEOM_ASSERT(Simple.Inside(outPt2 + dir2 * Dist) == vecgeom::EInside::kSurface);

  // DistanceToIn test for Inside points
  Dist = Simple.DistanceToIn(Vec_t(2., 0., 2.5), dir2);
  if (verbose) {
    std::cout << "DistanceToIn of inside Point (2.,0.,2.5) : (SHOULD BE NEGATIVE) : " << Dist << std::endl;
  }
  VECGEOM_ASSERT(Dist < 0.);

  Dist = Simple.DistanceToIn(Vec_t(1.5, 0., 1.99999), dir2);
  if (verbose) {
    std::cout << "DistanceToIn of inside Point (1.5,0.,1.99999) : (SHOULD BE NEGATIVE) : " << Dist << std::endl;
  }
  VECGEOM_ASSERT(Dist < 0.);

  Vec_t outPt3(6., 0., 5.);
  Vec_t dir3(Vec_t(5., 0., 3.) - outPt3);
  dir3.Normalize();
  Dist = Simple.DistanceToIn(outPt3, dir3);
  VECGEOM_ASSERT(Simple.Inside(outPt3 + dir3 * Dist) == vecgeom::EInside::kSurface);

  // DistanceToOut tests
  VECGEOM_ASSERT(Simple.DistanceToOut(Vec_t(3., 0., 2.5), Vec_t(0., 0., 1.)) == 0.5);
  VECGEOM_ASSERT(Simple.DistanceToOut(Vec_t(3., 0., 2.5), Vec_t(0., 0., -1.)) == 0.5);
  VECGEOM_ASSERT(Simple.DistanceToOut(Vec_t(5., 0., 2.5), Vec_t(1., 0., 0.)) == 0.);
  VECGEOM_ASSERT(Simple.DistanceToOut(Vec_t(5., 0., 2.5), Vec_t(-1., 0., 0.)) == 4.);
  VECGEOM_ASSERT(Simple.DistanceToOut(Vec_t(4.5, 0., 2.), Vec_t(0., 0., 1.)) == 1.);
  VECGEOM_ASSERT(Simple.DistanceToOut(Vec_t(4.5, 0., 1.9), Vec_t(0., 0., 1.)) == 1.1);
  VECGEOM_ASSERT(Simple.DistanceToOut(outPt1, Vec_t(0., 0., 1.)) < 0.);
  VECGEOM_ASSERT(Simple.DistanceToOut(outPt2, Vec_t(0., 0., 1.)) < 0.);
  VECGEOM_ASSERT(Simple.DistanceToOut(outPt3, Vec_t(0., 0., 1.)) < 0.);
#if (0)
  {
    /* This block contains some test points which shows some mismatches and detected by
     * Benchmarker BEFORE application of THE FIXES IN THE CONE
     *
     * Now all their results are expected and matches with G4
     *
     * GenericPolycone as specified in jira issue 425
     */

    Precision sphi         = 0.;
    Precision dphi         = vecgeom::kTwoPi;
    const int numRZ1       = 10;
    Precision polycone_r[] = {1, 5, 3, 4, 9, 9, 3, 3, 2, 1};
    Precision polycone_z[] = {0, 1, 2, 3, 0, 5, 4, 3, 2, 1};
    auto poly2             = new GenericPolycone_t("GenericPoly", sphi, dphi, numRZ1, polycone_r, polycone_z);

    // Some mismatched point detected by Benchmarker before fixes of Cones
    Vec_t ptIn(1.342000296513003121390284, 7.419050450955836595312576, 0.8763312698342673456863849);
    Vec_t dir(0.2092127200968610933884406, 0.7654562532663932161725029, 0.6085283576671245420186551);
    Dist          = poly2->DistanceToOut(ptIn, dir);
    Vec_t MovedPt = (ptIn + Dist * dir);
    if (verbose) {
      std::cout << "Location : " << poly2->Inside(ptIn) << std::endl;
      std::cout << "Calculated DistToOut : " << Dist << std::endl;
      std::cout << "Geant4     DistToOut : " << Dist << std::endl;
      std::cout << "Moved POInt : " << MovedPt << std::endl;
      std::cout << "Radius of Moved Point : " << MovedPt.Perp() << std::endl;
      std::cout << "Location of Moved Point : " << poly2->Inside(MovedPt) << std::endl;
    }

    Vec_t safetyPt(2.53568647643333200392135, 7.381743158896469481078384, 0.9386758016241848467942077);
    std::cout << "location of SafetyPoint : " << poly2->Inside(safetyPt) << std::endl;
    std::cout << "SafetyToOut : " << poly2->SafetyToOut(safetyPt) << std::endl;

    // Safety Distance mismatch (for Surface points) detected by shapetester
    Vec_t safetyPt2(-8.992832946941797800377572, -0.3591054026977471558268462, 0.5);
    Vec_t dirSafetyPt2(0.0760655128324308066334325, -0.2131854768574358571786576, 0.9740461951132538542807993);
    std::cout << "location of SafetyPoint2 : " << poly2->Inside(safetyPt2) << std::endl;
    std::cout << "SafetyToIn Dist : " << poly2->SafetyToIn(safetyPt2) << std::endl;
    std::cout << "DistanceToIn Dist : " << poly2->DistanceToIn(safetyPt2, dirSafetyPt2) << std::endl;

    Vec_t safetyPt3(-6.28455157823346866052816, -6.442391799981537658936759, 0.1422521521363713237207094);
    Vec_t dirSafetyPt3(-0.04592037192788539501364653, 0.3504705316940292525451639, 0.9354473399695512059182079);
    std::cout << "location of SafetyPoint3 : " << poly2->Inside(safetyPt3) << std::endl;
    std::cout << "DistanceToOut Dist : " << poly2->DistanceToOut(safetyPt3, dirSafetyPt3) << std::endl;

    std::cout << "=========================================================" << std::endl;
    ptIn.Set(-42.046543375136415932047384558246, 2.876852248775723985829699813621, 41.956705401227381457829324062914);
    dir.Set(0.712814657374801763367599960475, -0.207692873805265298958744324409, -0.669894718894061602654232956411);

    Dist = poly2->DistanceToOut(ptIn, dir);
    std::cout << "Location of Actual point : " << poly2->Inside(ptIn) << std::endl;
    std::cout << "DistancetoOut : 		" << Dist << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptIn + Dist * dir) << std::endl;
    std::cout << "Moved Point : 		" << (ptIn + Dist * dir) << std::endl;
    Dist = poly2->DistanceToIn(ptIn, dir);
    std::cout << "DistancetoIn : 		" << Dist << std::endl;
    std::cout << "Moved Point : 		" << (ptIn + Dist * dir) << std::endl;
    std::cout << "Radius of Moved Point : " << (ptIn + Dist * dir).Perp() << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptIn + Dist * dir) << std::endl;

    std::cout << "===========================================================" << std::endl;

    ptIn.Set(10., 0., 4.895);
    Vec_t tempPt(0., 0., 0.);
    Vec_t dirtemp = tempPt - ptIn;
    dir           = dirtemp.Unit();

    std::cout << "Location of Actual point : " << poly2->Inside(ptIn) << std::endl;
    Dist = poly2->DistanceToIn(ptIn, dir);
    std::cout << "DistancetoIn : 		" << Dist << std::endl;
    std::cout << "Moved Point : 		" << (ptIn + Dist * dir) << std::endl;
    std::cout << "Radius of Moved Point : " << (ptIn + Dist * dir).Perp() << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptIn + Dist * dir) << std::endl;

    std::cout << "===========================================================" << std::endl;
    Vec_t ptToIn(-29.402024989716856850918702548370, -28.599908958628695643255923641846,
                 -35.130380012379767151742271380499);
    Vec_t dirPtToIn(0.605765083102027368511244276306, 0.371690655275148829073117440203,
                    0.703487541379038683331259562692);
    Dist = poly2->DistanceToIn(ptToIn, dirPtToIn);
    std::cout << "DistancetoIn : 		" << Dist << std::endl;
    std::cout << "Moved Point : 		" << (ptToIn + Dist * dirPtToIn) << std::endl;
    std::cout << "Radius of Moved Point : " << (ptToIn + Dist * dirPtToIn).Perp() << std::endl;
    std::cout << "Location of Moved point : " << poly2->Inside(ptToIn + Dist * dirPtToIn) << std::endl;

    {
      std::cout << "\n======== Dissecting safetytoout ==================" << std::endl;
      Vec_t pt(-0.5731061999726463351834127, -8.981734202455164961520495, 0.3047303633861303540086851);
      Vec_t dir(-0.6122635507109631669564465, 0.3466338756867177739451336, -0.7106182524374172748693468);
      std::cout << pt << std::endl;
      std::cout << "Location : " << poly2->Inside(pt) << std::endl;
      Dist = poly2->DistanceToOut(pt, dir);
      std::cout << "DistanceToOut : " << Dist << std::endl;
      Dist = poly2->SafetyToOut(pt);
      std::cout << "SafetyToOut : " << Dist << std::endl;
    }
  }
#endif

  return true;
}

int main(int argc, char *argv[])
{
  VECGEOM_ASSERT(TestGenericPolycone<vecgeom::SimpleGenericPolycone>());
  std::cout << "VecGeomGenericPolycone passed\n";

  return 0;
}
