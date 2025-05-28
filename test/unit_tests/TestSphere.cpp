//
//
// TestSphere
//             Ensure asserts are compiled in

// ensure asserts are compiled in
#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Sphere.h"
#include "ApproxEqual.h"

#include <cmath>
#include <iomanip>

#define PI 3.14159265358979323846
#define deg PI / 180.

using vecgeom::kInfLength;
using vecgeom::Precision;

bool TestWrongSidePoint(double Dist, std::string msg)
{
  bool verbose = false;
  if (verbose) std::cout << msg << " : " << Dist << std::endl;
  return (Dist <= 0.);
}

template <class Sphere_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision>>
bool TestSphere()
{
  //  int verbose  = 1;
  double fRmin = 0, fRmax = 3, fSPhi = 0, fDPhi = 2 * PI, fSTheta = 0, fDTheta = PI;
  double fR = fRmax;

  double Dist;
  Vec_t pzero(0, 0, 0);
  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);

  Sphere_t b1("Solid VecGeomSphere #1", fRmin, fRmax, fSPhi, fDPhi, fSTheta, fDTheta);
  Sphere_t b2("Solid VecGeomSphere #2", fRmin, fRmax, PI / 6, PI / 2, fSTheta, fDTheta);
  Sphere_t b3("Solid VecGeomSphere #3", 1, fRmax, PI / 6, PI / 2, fSTheta, fDTheta);

  // Check cubic volume
  VECGEOM_ASSERT(ApproxEqual<Precision>(b1.Capacity(), ((4 * PI / 3) * fR * fR * fR)));
  VECGEOM_ASSERT(ApproxEqual<Precision>(b2.Capacity(), (((4 * PI / 3) * fR * fR * fR) / 4)));
  // std::cout<<"Capacity of B3 : "<<b3.Capacity()<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(b3.Capacity(), 27.22713633111));

  // std::cout<<"Capacity : "<<b2.Capacity()<<std::endl;
  // Check Surface area
  VECGEOM_ASSERT(ApproxEqual<Precision>(b1.SurfaceArea(), ((4 * PI) * fR * fR)));
  VECGEOM_ASSERT(ApproxEqual<Precision>(b2.SurfaceArea(), ((((4 * PI) * fR * fR) / 4) + (PI * fR * fR))));
  VECGEOM_ASSERT(ApproxEqual<Precision>(b2.SurfaceArea(), 56.5486677646));
  // std::cout<<std::setprecision(12)<<"SurfaceArea of B3 : "<< b3.SurfaceArea() << std::endl;
  // std::cout<<"Not : "<< b1.SurfaceArea()/4 << std::endl;

  // Check Extent and cached BBox
  Vec_t minExtent, maxExtent;
  Vec_t minBBox, maxBBox;
  b1.Extent(minExtent, maxExtent);
  b1.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-fR, -fR, -fR)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(fR, fR, fR)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));
  b2.Extent(minExtent, maxExtent);
  b2.GetUnplacedVolume()->GetBBox(minBBox, maxBBox);
  VECGEOM_ASSERT(ApproxEqual(minExtent, Vec_t(-1.5, 0, -3.)));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, Vec_t(2.59808, 3., 3.)));
  VECGEOM_ASSERT(ApproxEqual(minExtent, minBBox));
  VECGEOM_ASSERT(ApproxEqual(maxExtent, maxBBox));
  // VECGEOM_ASSERT(ApproxEqual(minExtent,Vec_t(-6,-6,-6)));
  // VECGEOM_ASSERT(ApproxEqual(maxExtent,Vec_t( 6, 6, 6)));

  bool valid;
  Vec_t ponx(8, 0., 0.), pony(0., 8., 0.), ponz(0., 0., 8.);
  Vec_t ponmx(-8, 0., 0.), ponmy(0., -8., 0.), ponmz(0., 0., -8.);
  Vec_t normal;

  //______________________________________________________________________________________

  // Sphere for testing Real Important functions like Inside, Saftey , DistanceToIn, DistanceToOut

  // Tests considering fullPhiSphere and fullThetaSphere

  Sphere_t b4("Solid VecGeomSphere #4", 6, 8, fSPhi, fDPhi, fSTheta, fDTheta);

  // Completely Inside Point
  Vec_t pointI(0.337317535387103, 6.265557026812151,
               3.103064940359008); // Point at the distance of 7 from center of sphere i.e. completely inside point
  VECGEOM_ASSERT(b4.Inside(pointI) == vecgeom::EInside::kInside); // Should pass

  // Completely Outside Point
  Vec_t pointO(0.162722224118177, -8.624868144449298,
               -2.566158796458579); // Point at the distance of 9 from center of sphere i.e. completely outside point
  VECGEOM_ASSERT(b4.Inside(pointO) == vecgeom::EInside::kOutside); // Should pass

  // Point very very very close to fRmax should be considered on the surface
  Vec_t pointOS(-2.393620784732096, -4.753290873255242,
                5.973006396542139); // Point at the distance of 8.0000000000004 from center of sphere i.e. Surface point
  VECGEOM_ASSERT(b4.Inside(pointOS) == vecgeom::EInside::kSurface); // Should pass

  // Point very very very close to fRmin should be considered on the surface
  Vec_t pointIS(
      -4.231374871926247, -4.183095200116632,
      -0.772775025466741); // Point at the distance of 5.9999999999994 from center of sphere i.e. Surface point
  VECGEOM_ASSERT(b4.Inside(pointIS) == vecgeom::EInside::kSurface); // Should pass

  // Point Just inside the inner tolerance of fRmax should be considered as inside point
  Vec_t pointJIO(5.644948390009752, -5.435958227995553, 1.607767309536356); // Point at the distance of 7.999999994 from
                                                                            // center of sphere i.e. completely inside
                                                                            // point considering tolerance limit is 1e-9
  // VECGEOM_ASSERT(b4.Inside(pointJIO)==vecgeom::EInside::kInside); //Should pass

  // Point Just outside the inner tolerance of fRmin should be considered as inside point
  Vec_t pointJOI(0.101529841379922, -5.993375366532538, 0.262951431162379); // Point at the distance of 6.000000004 from
                                                                            // center of sphere i.e. completely inside
                                                                            // point considering tolerance limit is 1e-9
  // VECGEOM_ASSERT(b4.Inside(pointJOI)==vecgeom::EInside::kInside); //Should pass

  // Testing Safety Functions
  Dist = b4.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - 8));

  // SafetyToOut return -1 if point is outside
  Dist = b4.SafetyToOut(pzero);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b4.SafetyToOut(pointO); // using outside point for b4
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Vec_t pointO2(9, 0, 0);
  Dist = b4.SafetyToOut(pointO2); // using outside point for b4
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Vec_t pointO3(0, 9, 0);
  Dist = b4.SafetyToOut(pointO3); // using outside point for b4
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Vec_t pointO4(0, 0, 9);
  Dist = b4.SafetyToOut(pointO4); // using outside point for b4
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Vec_t pointBWRminRmaxXI(6.5, 0, 0);
  Dist = b4.SafetyToOut(pointBWRminRmaxXI);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.5));

  Vec_t pointBWRminRmaxYI(0, 6.5, 0);
  Dist = b4.SafetyToOut(pointBWRminRmaxYI);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.5));

  Vec_t pointBWRminRmaxZI(0, 0, 6.5);
  Dist = b4.SafetyToOut(pointBWRminRmaxZI);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.5));

  Vec_t pointBWRminRmaxXO(7.5, 0, 0);
  Dist = b4.SafetyToOut(pointBWRminRmaxXO);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.5));

  Vec_t pointBWRminRmaxYO(0, 7.5, 0);
  Dist = b4.SafetyToOut(pointBWRminRmaxYO);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.5));

  Vec_t pointBWRminRmaxZO(0, 0, 7.5);
  Dist = b4.SafetyToOut(pointBWRminRmaxZO);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.5));

  // For Inside point SafetyToIn should returns -1
  Dist = b4.SafetyToIn(pointBWRminRmaxXO);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToIn for inside point"));

  Dist = b4.SafetyToIn(pointBWRminRmaxYO);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToIn for inside point"));

  Dist = b4.SafetyToIn(pointBWRminRmaxZO);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToIn for inside point"));

  Vec_t genPointBWRminRmax(3.796560684305335, -6.207283535497058,
                           2.519078815824183); // Point at distance of 7.7 from center. i.e. inside point,
                                               // SafetyToIn should return -1
  Dist = b4.SafetyToIn(genPointBWRminRmax);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToIn for inside point"));

  valid = b4.Normal(ponx, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(1, 0, 0)));
  VECGEOM_ASSERT(valid);
  valid = b4.Normal(pony, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 1, 0)));
  VECGEOM_ASSERT(valid);
  valid = b4.Normal(ponz, normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, 1)));
  valid = b4.Normal(ponmx, normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-1, 0, 0)));
  valid = b4.Normal(ponmy, normal);
  VECGEOM_ASSERT(valid);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, -1, 0)));
  valid = b4.Normal(ponmz, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0, 0, -1)));
  VECGEOM_ASSERT(valid);

  //________________________________________________________________________________________________________________
  // Sphere for testing Real Important functions like Inside, Saftey , DistanceToIn, DistanceToOut
  // Considering !FullPhiSphere but FullThetaSphere
  Sphere_t b5("Solid VecGeomSphere #5", 6, 8, fSPhi, fDPhi / 2, fSTheta, fDTheta);
  // Completely Inside section of a sphere from 0-PI, but radially inside
  Vec_t pointISec_0_PI(-6.354024677915919, 2.724956501738179, -1.095893451057325); // Point at the distance of 7 from
                                                                                   // center of sphere and in 0-PI Phi
                                                                                   // range i.e. completely inside point
  VECGEOM_ASSERT(b5.Inside(pointISec_0_PI) == vecgeom::EInside::kInside);          // Should pass

  // Completely Inside section PI-2PI but completely outside section 0-PI, but radially inside
  // Vec_t pointISec_PI_2PI(4.686901690014317,-1.144935913500135,5.071693435344703); //Point at the distance of 7 from
  // center of sphere and in PI-2PI Phi range i.e. completely outside
  Vec_t pointISec_PI_2PI(-2.523183636323228, -3.188902592924771, -5.697757856405303);
  VECGEOM_ASSERT(b5.Inside(pointISec_PI_2PI) == vecgeom::EInside::kOutside); // Should pass
  // std::cout<<"PI - 2PI point line 115 : "<<b5.Inside(pointISec_PI_2PI)<<std::endl;

  // Point very very very close to end Phi angle (fSPhi+fDPhi) should be considered on the surface
  Vec_t pointISurface_Sec_0_PI(-6.975035234148935, 0.000000000583791,
                               -0.590663594934467); // Point at the distance of 7 from center of sphere and very very
                                                    // very close to PI in the inside region in 0-PI Phi range
  // i.e.  radially inside for fullPhiSphere but for spherical section of 0-PI it should be on surface
  VECGEOM_ASSERT(b5.Inside(pointISurface_Sec_0_PI) == vecgeom::EInside::kSurface); // Should pass

  Vec_t pointOSurface_Sec_0_PI(-6.859517189252706, -0.000000002746520,
                               1.395357993615493); // Point at the distance of 7 from center of sphere and very very
                                                   // very close to PI in the outside region in 0-PI Phi range
  // i.e.  radially inside for fullPhiSphere but for spherical section of 0-PI it should be on surface
  // VECGEOM_ASSERT(b5.Inside(pointOSurface_Sec_0_PI)==vecgeom::EInside::kSurface); //Should pass

  // Considering !FullPhiSphere but FullThetaSphere
  Sphere_t b6("Solid VecGeomSphere #6", 6, 8, PI / 6, PI / 3, fSTheta, fDTheta); // Spherical section from 30-60 degree
  Vec_t pointI_30_60(5.343263338886987, 3.536603816131425, -2.818150162612158);  // Point at the distance of 7 from
                                                                                 // center of sphere and in 30-60 Phi
                                                                                 // range i.e. completely inside
  VECGEOM_ASSERT(b6.Inside(pointI_30_60) == vecgeom::EInside::kInside);          // Should pass

  Vec_t pointO_30_60(6.043467293054153, 0.461937628581358, 3.501873313683026); ////Point at the distance of 7 from
  /// center of sphere and in 0-30 Phi range
  /// i.e. completely outside
  VECGEOM_ASSERT(b6.Inside(pointO_30_60) == vecgeom::EInside::kOutside); // Should pass

  Vec_t pointISurface_Sec_30_60(0.139868946641446, 0.080753374000628, 6.998136578429498);
  VECGEOM_ASSERT(b6.Inside(pointISurface_Sec_30_60) == vecgeom::EInside::kSurface); // Should pass

  Vec_t pointOSurface_Sec_30_60(4.589551141500899, 2.649778586424563,
                                4.573258549707597); // Point at the distance of 7 from center of sphere and very very
                                                    // very close to 30 in the outside region in 30-60 Phi range
  VECGEOM_ASSERT(b6.Inside(pointOSurface_Sec_30_60) == vecgeom::EInside::kSurface); // Should pass

  Sphere_t b8("Solid VecGeomSphere #8", 6, 8, PI / 6, PI / 6, fSTheta,
              fDTheta); // Spherical section from 30-60 degree in PHI and 0-180 in THETA
  Vec_t pointOSafety_phi_30_60_theta_0_180(0, 5, 0);
  Dist = b8.SafetyToIn(pointOSafety_phi_30_60_theta_0_180);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 2.50000));

  Dist = b8.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b8.SafetyToIn(pbigy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b8.SafetyToIn(pbigz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b8.SafetyToIn(pbigmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b8.SafetyToIn(pbigmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b8.SafetyToIn(pbigmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));

  Dist = b8.SafetyToOut(pbigx);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b8.SafetyToOut(pbigy);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b8.SafetyToOut(pbigz);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b8.SafetyToOut(pbigmx);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b8.SafetyToOut(pbigmy);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b8.SafetyToOut(pbigmz);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  // Point between Rmin and Rmax but outside Phi Range
  Vec_t point_phi_60_90(0.041433182037376, 3.902131470711286,
                        -6.637895244481554); // Point at a distance of 7.7 from center and in 60-90 phi range. i.e
                                             // completely outside point. DistanceFromInside shoudl return zero
  Dist = b8.SafetyToOut(point_phi_60_90);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b8.SafetyToIn(point_phi_60_90);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 1.91518354715165));

  // Checking NORMAL FUNCTION
  // Sphere in 60-120 Phi range
  Sphere_t b10("Solid VecGeomSphere #10", 6, 8, PI / 3, PI / 3, fSTheta, fDTheta);
  Vec_t point_phi_60_120_rad_9(2.821923096621196, 6.085843652625694, 5.999938089059868);
  valid = b10.Normal(point_phi_60_120_rad_9, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0.866025, -0.5, 0))); // Verified with Geant4
  //________________________________________________________________________________________________________________

  // Tests Considering !FullPhiSphere and !FullThetaSphere
  Sphere_t b7("Solid VecGeomSphere #7", 6, 8, PI / 6, PI / 6, 0.,
              PI / 6); // Spherical section from 30-60 degree in PHI and 0-30 in THETA

  Vec_t pointI_phi_30_60_theta_0_30(
      2.051950506806283, 1.301790637821730,
      6.564666042754737); // Point at the distance of 7 from center of sphere and in 30-60 Phi range and 0-30 in Theta
  VECGEOM_ASSERT(b7.Inside(pointI_phi_30_60_theta_0_30) == vecgeom::EInside::kInside); // Should pass

  Vec_t pointO_phi_30_60_theta_31_45(3.038887886090341, 3.240358815458723,
                                     5.409735221140894); // Point at the distance of 7 from center of sphere and in
                                                         // 30-60 Phi range and 31-45 in Theta; completely outside in
                                                         // terms of theta
  VECGEOM_ASSERT(b7.Inside(pointO_phi_30_60_theta_31_45) == vecgeom::EInside::kOutside); // Should pass

  Vec_t pointOSurface_phi_30_60_theta_0_30(2.421058285775014, 2.527543625149363,
                                           6.062177826478448); // Point at the distance of 7 from center of sphere and
                                                               // very very very close to 30 in the outside region in
                                                               // 0-30 theta range
  // i.e.  radially inside, also inside in terms of phi (30-60) but for theta section of 0-30 it should be on surface
  VECGEOM_ASSERT(b7.Inside(pointOSurface_phi_30_60_theta_0_30) == vecgeom::EInside::kSurface); // Should pass

  Vec_t pointISurface_phi_30_60_theta_0_30(2.983684745081379, 1.829651698111631,
                                           6.062177826950138); // Point at the distance of 7 from center of sphere and
                                                               // very very very close to 30 in the inside region in
                                                               // 0-30 theta range
  // i.e.  radially inside, also inside in terms of phi (30-60) but for theta section of 0-30 it should be on surface
  VECGEOM_ASSERT(b7.Inside(pointISurface_phi_30_60_theta_0_30) == vecgeom::EInside::kSurface); // Should pass

  Sphere_t b9("Solid VecGeomSphere #9", 6, 8, PI / 6, PI / 6, PI / 6, PI / 6);
  Dist = b9.SafetyToIn(pbigx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b9.SafetyToIn(pbigy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b9.SafetyToIn(pbigz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b9.SafetyToIn(pbigmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b9.SafetyToIn(pbigmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));
  Dist = b9.SafetyToIn(pbigmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 92));

  // For Outside point SafetyToOut return zero. Following test should pass
  Dist = b9.SafetyToOut(pbigx);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b9.SafetyToOut(pbigy);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b9.SafetyToOut(pbigz);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b9.SafetyToOut(pbigmx);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b9.SafetyToOut(pbigmy);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = b9.SafetyToOut(pbigmz);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  // For Completely inside point it should Dist should be zero. Using data of b7 sphere.
  Dist = b7.SafetyToIn(pointI_phi_30_60_theta_0_30);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToIn for inside point"));

  Dist = b7.SafetyToIn(pointO_phi_30_60_theta_31_45);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 1.142348737370068));

  valid = b7.Normal(pointO_phi_30_60_theta_31_45, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(-0.434127, -0.462908, -0.772819))); // Verified with Geant4

  //________________________________________________________________________________________________________________
  //
  // Add unit test from Geant4
  //

  Vec_t px(30, 0, 0), py(0, 30, 0), pz(0, 0, 30);
  Vec_t pmx(-30, 0, 0), pmy(0, -30, 0), pmz(0, 0, -30);
  Vec_t ponrmin1(45, 0, 0), ponrmax1(50, 0, 0), ponzmax(0, 0, 50), ponrmin2(45 / std::sqrt(2.), 45 / std::sqrt(2.), 0),
      ponrmin3(0, 0, -45), ponrminJ(0, 0, -300), ponrmaxJ(0, 0, -500),
      ponrmax2(50 / std::sqrt(2.), 50 / std::sqrt(2.), 0);
  Vec_t ponphi1(48 / std::sqrt(2.), -48 / std::sqrt(2.), 0), ponphi2(48 / std::sqrt(2.), 48 / std::sqrt(2.), 0),
      pInPhi(48 * 0.866, -24, 0), pOverPhi(-48 / std::sqrt(2.), 48 / std::sqrt(2.), 0);
  Vec_t pontheta1(0, 48 * std::sin(PI / 4), 48 * std::cos(PI / 4)),
      pontheta2(0, 48 * std::sin(PI / 4), -48 * std::cos(PI / 4));

  Vec_t ptestphi1(-100, -45 / std::sqrt(2.), 0), ptestphi2(-100, 45 / std::sqrt(2.), 0);

  Vec_t ptesttheta1(0, 48 / std::sqrt(2.), 100), ptesttheta2(0, 48 / std::sqrt(2.), -100);

  // Directions

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.), 1 / std::sqrt(2.), 0), vmxmy(-1 / std::sqrt(2.), -1 / std::sqrt(2.), 0);
  Vec_t vxmy(1 / std::sqrt(2.), -1 / std::sqrt(2.), 0), vmxy(-1 / std::sqrt(2.), 1 / std::sqrt(2.), 0);
  Vec_t vxmz(1 / std::sqrt(2.), 0., -1 / std::sqrt(2.)), vymz(0., 1 / std::sqrt(2.), -1 / std::sqrt(2.));
  Vec_t vmxmz(-1 / std::sqrt(2.), 0., -1 / std::sqrt(2.)), vmxz(-1 / std::sqrt(2.), 0., 1 / std::sqrt(2.));

  Vec_t v345exit1(-0.8, 0.6, 0), v345exit2(0.8, 0.6, 0), v345exit3(0.6, 0.8, 0);

  Vec_t pRand, vRand;
  vecgeom::EnumInside inside;
  Sphere_t s1("Solid Sphere_t", 0, 50, 0, 2 * PI, 0, PI);
  Sphere_t sn1("sn1", 0, 50, PI * 0.5, 3. * PI * 0.5, 0, PI);

  // Theta sections

  Sphere_t sn11("sn11", 0, 50, 0, 2 * PI, 0., 0.5 * PI);
  Sphere_t sn12("sn12", 0, 50, 0, 2 * PI, 0., 0.25 * PI);
  Sphere_t sn13("sn13", 0, 50, 0, 2 * PI, 0.75 * PI, 0.25 * PI);
  Sphere_t sn14("sn14", 0, 50, 0, 2 * PI, 0.25 * PI, 0.75 * PI);
  Sphere_t sn15("sn15", 0, 50, 0, 2 * PI, 89. * deg, 91. * deg);
  Sphere_t sn16("sn16", 0, 50, 0, 2 * PI, 0. * deg, 89. * deg);
  Sphere_t sn17("sn17", 0, 50, 0, 2 * PI, 91. * deg, 89. * deg);

  Sphere_t s2("Spherical Shell", 45, 50, 0, 2 * PI, 0, PI);
  Sphere_t sn2("sn2", 45, 50, 0.5 * PI, 0.5 * PI, 0, PI);
  Sphere_t sn22("sn22", 0, 50, 0.5 * PI, 0.5 * PI, 0, PI);

  Sphere_t s3("Band (theta segment)", 45, 50, 0, 2 * PI, PI / 4, 0.5 * PI);
  Sphere_t s32("Band (theta segment2)", 45, 50, 0, 2 * PI, 0, PI / 4);
  Sphere_t s33("Band (theta segment1)", 45, 50, 0, 2 * PI, PI * 3 / 4, PI / 4);
  Sphere_t s34("Band (theta segment)", 4, 50, 0, 2 * PI, PI / 4, 0.5 * PI);
  Sphere_t s4("Band (phi segment)", 45, 50, -PI / 4, 0.5 * PI, 0, PI);
  //    std::cout<<"s4.fSPhi = "<<s4.GetSPhi()<<std::endl;
  Sphere_t s41("Band (phi segment)", 5, 50, -PI, 3. * PI / 2., 0, PI);
  Sphere_t s42("Band (phi segment)", 5, 50, -PI / 2, 3. * PI / 2., 0, PI);
  Sphere_t s5("Patch (phi/theta seg)", 45, 50, -PI / 4, 0.5 * PI, PI / 4, 0.5 * PI);

  Sphere_t s6("John example", 300, 500, 0, 5.76, 0, PI);
  Sphere_t s7("sphere7", 1400., 1550., 0.022321428571428572, 0.014642857142857141, 1.5631177553663251,
              0.014642857142857141);
  Sphere_t s8("sphere", 278.746, 280.0, 0.0 * deg, 360.0 * deg, 0.0 * deg, 90.0 * deg);
  Sphere_t b216("b216", 1400.0, 1550.0, 0.022321428571428572, 0.014642857142857141, 1.578117755366325,
                0.014642857142867141);

  Sphere_t s9("s9", 0, 410, 0 * deg, 360 * deg, 90 * deg, 90 * deg);

  Sphere_t b402("b402", 475, 480, 0 * deg, 360 * deg, 17.8 * deg, 144.4 * deg);

  Sphere_t b1046("b1046", 4750 * 1e6, 4800 * 1e6, 0 * deg, 360 * deg, 0 * deg, 90 * deg);

  Sphere_t testTheta("opticalEscape", 0., 2.995, 0 * deg, 360 * deg, 0 * deg, 120 * deg);

  Vec_t p402(471.7356120367253, 51.95081450791341, 5.938043020529463);

  Vec_t v402(-0.519985502840818, 0.2521089719986221, 0.8161226274728446);

  Vec_t p216(1549.9518578505142, 1.2195415370970153, -12.155289555510985),
      v216(-0.61254821852534425, -0.51164551429243466, -0.60249775741147549);

  Vec_t s9p(384.8213314370455, 134.264386151667, -44.56026800002064);

  Vec_t s9v(-0.6542770611918751, -0.0695116921641141, -0.7530535517814154);

  Sphere_t s10("s10", 0, 0.018, 0 * deg, 360 * deg, 0 * deg, 180 * deg);

  Vec_t s10p(0.01160957408065766, 0.01308205826682229, 0.004293345210644617);

  Sphere_t s11("s11", 5000000., 3700000000., 0 * deg, 360 * deg, 0 * deg, 180 * deg);

  Vec_t ps11(-3072559844.81995153427124, -1924000000, -740000000);

  Sphere_t sAlex("sAlex", 500., 501., 0 * deg, 360 * deg, 0 * deg, 180 * deg);

  Vec_t psAlex(-360.4617031263808, -158.1198807105035, 308.326878333183);

  Vec_t vsAlex(-0.7360912456240805, -0.4955800202572754, 0.4610532741813497);

  Sphere_t sLHCB("sLHCB", 8600, 8606, -1.699135525184141 * deg, 3.398271050368263 * deg, 88.52855940538514 * deg,
                 2.942881189229715 * deg);

  Vec_t pLHCB(8600.242072535835, -255.1193517702246, -69.0010277128286);

  Sphere_t b658("b658", 209.6, 211.2658, 0.0 * deg, 360 * deg, 0.0 * deg, 90 * deg);

  Vec_t p658(-35.69953348982516, 198.3183279249958, 56.30959457033987);
  Vec_t v658(-.2346058124516908, -0.9450502890785083, 0.2276841318065671);

  Sphere_t spAroundX("SpAroundX", 10., 1000., -1.0 * deg, 2.0 * deg, 0. * deg, 180.0 * deg);

  double radOne    = 100.0;
  double angle     = -1.0 * deg - 0.25 * vecgeom::kTolerance * 1000;
  Vec_t ptPhiMinus = Vec_t(radOne * std::cos(angle), radOne * std::sin(angle), 0.0);

  // spheres for theta cone intersections

  Sphere_t s13("Band (theta segment)", 5, 50, 0, 2 * PI, PI / 6., 0.5 * PI);
  Sphere_t s14("Band (theta segment)", 5, 50, 0, 2 * PI, PI / 3., 0.5 * PI);

  // b. 830

  double mainInnerRadius = 21.45 * 10;
  double mainOuterRadius = 85.0 * 10;
  double minTheta        = 18.0 * deg;

  Sphere_t sb830("mainSp", mainInnerRadius, mainOuterRadius, 0.0, M_PI * 2, minTheta, M_PI - 2 * minTheta);

  Vec_t pb830(81.61117212, -27.77179755, 196.4143423);
  Vec_t vb830(0.1644697995, 0.18507236, 0.9688642354);

  // Check Sphere_t::Inside

  inside = s7.Inside(Vec_t(1399.984667238032, 5.9396696802500299, -2.7661927818688308));
  // std::cout<<"s7.Inside(Vec_t(1399.98466 ... = "
  //       <<inside<<std::endl ;

  inside = s8.Inside(Vec_t(-249.5020724528353, 26.81253142743162, 114.8988524453591));
  // std::cout<<"s8.Inside(Vec_t(-249.5020 ... = "<<inside<<std::endl ;
  inside = b216.Inside(p216);
  // std::cout<<"b216.Inside(p216) = "<<inside<<std::endl ;
  inside = s1.Inside(pz);
  // std::cout<<"s1.Inside(pz) = "<<inside<<std::endl ;

  inside = s9.Inside(s9p);
  // std::cout<<"s9.Inside(s9p) = "<<inside<<std::endl ;

  inside = b402.Inside(p402);
  // std::cout<<"p402.Inside(p402) = "<<nside<<std::endl ;

  inside = s10.Inside(s10p);
  // std::cout<<"s10.Inside(s10p) = "<<inside<<std::endl ;
  // std::cout<<"p radius = "<<s10p.mag()<<std::endl ;

  inside = s11.Inside(ps11);
  // std::cout<<"s11.Inside(ps11) = "<<inside<<std::endl ;
  // std::cout<<"ps11.mag() = "<<ps11.mag()<<std::endl ;

  inside = sLHCB.Inside(pLHCB);
  // std::cout<<"sLHCB.Inside(pLHCB) = "<<inside<<std::endl ;
  // std::cout<<"pLHCB.mag() = "<<pLHCB.mag()<<std::endl ;

  inside = spAroundX.Inside(ptPhiMinus);
  // std::cout<<"spAroundX.Inside(ptPhiMinus) = "<<inside<<std::endl ;
  // VECGEOM_ASSERT(inside==vecgeom::EInside::kSurface);
  inside = b658.Inside(p658);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kOutside);
  // std::cout<<"b658.Inside(p658) = "<<inside<<std::endl ;

  VECGEOM_ASSERT(s1.Inside(pzero) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(s2.Inside(pzero) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(s2.Inside(ponrmin2) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(s2.Inside(ponrmax2) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(s3.Inside(pontheta1) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(s3.Inside(pontheta2) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(s4.Inside(ponphi1) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(s4.Inside(ponphi1) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(s4.Inside(pOverPhi) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(s4.Inside(pInPhi) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(s5.Inside(pbigz) == vecgeom::EInside::kOutside);

  VECGEOM_ASSERT(s41.Inside(pmx) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(s42.Inside(pmx) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(sn11.Inside(pzero) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(sn12.Inside(pzero) == vecgeom::EInside::kSurface);
  // std::cout<<"sn11.Inside0="<<sn11.Inside(pzero)<<std::endl;
  // std::cout<<"sn12.Inside0="<<sn12.Inside(pzero)<<std::endl;
  // std::cout<<"sn13.Inside0="<<sn13.Inside(pzero)<<std::endl;
  // std::cout<<"sn13.InsideZ="<<sn13.Inside(Vec_t(0,0,1.))<<std::endl;
  VECGEOM_ASSERT(sn13.Inside(pzero) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(sn13.Inside(-pz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(sn12.Inside(pz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(sn12.Inside(px) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(sn11.Inside(px) == vecgeom::EInside::kSurface);

  // Checking Sphere_t::SurfaceNormal
  double p2 = 1. / std::sqrt(2.), p3 = 1. / std::sqrt(3.);

  valid = sn1.Normal(Vec_t(0., 0., 50.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p3, p3, p3)));
  valid = sn1.Normal(Vec_t(0., 0., 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p2, p2, 0.)));

  valid = sn11.Normal(Vec_t(0., 0., 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0., 0., -1.)));
  valid = sn12.Normal(Vec_t(0., 0., 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0., 0., -1.)));
  valid = sn13.Normal(Vec_t(0., 0., 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(0., 0., 1.)));

  valid = sn2.Normal(Vec_t(-45., 0., 0.), normal);
  VECGEOM_ASSERT(ApproxEqual(normal, Vec_t(p2, -p2, 0.)));

  valid = s1.Normal(ponrmax1, normal);
  VECGEOM_ASSERT(ApproxEqual(normal, vx));

  // Checking Sphere_t::SafetyToOut(P)
  // Full Sphere
  Dist = s1.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = s1.SafetyToOut(ponrmax1);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = s1.SafetyToOut(px);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  // Full Sphere with Rmin
  Dist = s2.SafetyToOut(pzero);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = s2.SafetyToOut(ponrmax1);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = s2.SafetyToOut(ponrmin1);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  // Sphere with full Phi and Theta section
  Dist = sn11.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = sn11.SafetyToOut(px);
  // std::cout<<"Dist=sn11.SafetyToOut(px) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = sn12.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = sn12.SafetyToOut(pz);
  // std::cout<<"Dist=sn12.Inside(pz) = "<<sn12.Inside(pz)<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = sn13.SafetyToOut(-pz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20));
  Dist = s3.SafetyToOut(pzero);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  // std::cout<<"Dist=sn3.pzero = "<<Dist<<std::endl;

  Dist = s9.SafetyToOut(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  Dist = s9.SafetyToOut(px);
  // std::cout<<"Dist=sn11.SafetyToOut(px) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));
  // Sphere with Phi section
  Dist = s41.SafetyToOut(pzero);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "SafetyToOut for outside point"));

  Dist = s41.SafetyToOut(ponrmax1);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  // Checking Sphere_t::DistanceToOut(p,v)

  Dist = s1.DistanceToOut(pz, vz);
  // std::cout<<"Dist=s1.DistanceToOut(pz,vz) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20.));

  Dist  = s1.DistanceToOut(ponrmax1, vx);
  valid = s1.Normal(ponrmax1 + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vx));

  Dist = s2.DistanceToOut(ponrmin1, vx);
  // std::cout<<"Dist : "<<Dist<<" :: Normal : "<<norm<<std::endl;
  valid = s2.Normal(ponrmin1 + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 5) && ApproxEqual(normal, vx));

  Dist  = s2.DistanceToOut(ponrmax2, vx);
  valid = s2.Normal(ponrmax2 + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0) && ApproxEqual(normal, vxy));

  Dist  = s1.DistanceToOut(pzero, vx);
  valid = s1.Normal(pzero + Dist * vx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vx));
  Dist  = s1.DistanceToOut(pzero, vmx);
  valid = s1.Normal(pzero + Dist * vmx, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vmx));
  Dist  = s1.DistanceToOut(pzero, vy);
  valid = s1.Normal(pzero + Dist * vy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vy));
  Dist  = s1.DistanceToOut(pzero, vmy);
  valid = s1.Normal(pzero + Dist * vmy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vmy));
  Dist  = s1.DistanceToOut(pzero, vz);
  valid = s1.Normal(pzero + Dist * vz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vz));
  Dist  = s1.DistanceToOut(pzero, vmz);
  valid = s1.Normal(pzero + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vmz));
  Dist  = s1.DistanceToOut(pzero, vxy);
  valid = s1.Normal(pzero + Dist * vxy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vxy));

  Dist  = s4.DistanceToOut(ponphi1, vx);
  valid = s4.Normal(ponphi1 + Dist * vx, normal);
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,0)&&ApproxEqual(normal,vmxmy));
  Dist = s4.DistanceToOut(ponphi2, vx);
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,0)&&ApproxEqual(normal,vmxy));
  Dist = s3.DistanceToOut(pontheta1, vz);
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,0)&&ApproxEqual(normal,vy));
  Dist = s32.DistanceToOut(pontheta1, vmz);
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,0)&&ApproxEqual(normal,vmy));
  Dist = s32.DistanceToOut(pontheta1, vz);
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,50)&&ApproxEqual(normal,vz));
  Dist  = s1.DistanceToOut(pzero, vmz);
  valid = s1.Normal(pzero + Dist * vmz, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vmz));
  Dist  = s1.DistanceToOut(pzero, vxy);
  valid = s1.Normal(pzero + Dist * vxy, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50) && ApproxEqual(normal, vxy));

  Dist = s2.DistanceToOut(ponrmin1, vxy);
  //    std::cout<<"Dist=s2.DistanceToOut(pormin1,vxy) = "<<Dist<<std::endl;

  Dist = s2.DistanceToOut(ponrmax1, vmx);
  //    std::cout<<"Dist=s2.DistanceToOut(ponxside,vmx) = "<<Dist<<std::endl;
  Dist = s2.DistanceToOut(ponrmax1, vmxmy);
  //    std::cout<<"Dist=s2.DistanceToOut(ponxside,vmxmy) = "<<Dist<<std::endl;
  Dist = s2.DistanceToOut(ponrmax1, vz);
  //    std::cout<<"Dist=s2.DistanceToOut(ponxside,vz) = "<<Dist<<std::endl;

  Dist = s2.DistanceToOut(pbigx, vx);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "DistanceToOut for Outside point: "));

  Dist = s2.DistanceToOut(pbigx, vmx);
  //  VECGEOM_ASSERT(Dist < 0.);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "DistanceToOut for Outside point: "));

  Dist = s2.DistanceToOut(pbigy, vy);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "DistanceToOut for Outside point: "));

  Dist = s2.DistanceToOut(pbigy, vmy);
  VECGEOM_ASSERT(TestWrongSidePoint(Dist, "DistanceToOut for Outside point: "));

  // Test Distance for phi section
  // This test case is for the point grazing on the Wedge surface
  Dist = sn22.DistanceToOut(Vec_t(0., 49., 0.), vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 49.));

  Dist = sn22.DistanceToOut(Vec_t(-45., 0., 0.), vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 45.));
  // std::cout<<"Dist from Center ="<<sn22.DistanceToOut(Vec_t(0.,49.,0),Vec_t(0,-1,0),normal,convex)<<std::endl;
  // std::cout<<"Dist from Center ="<<sn22.DistanceToOut(Vec_t(-45.,0.,0),Vec_t(1,0,0),normal,convex)<<std::endl;

  Dist = s13.DistanceToOut(Vec_t(20., 0., 0.), vz);
  // std::cout<<"s13.DistanceToOut(Vec_t(20.,0.,0.),vz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 34.641016151377549));

  Dist = s13.DistanceToOut(Vec_t(20., 0., 0.), vmz);
  // std::cout<<"s13.DistanceToOut(Vec_t(20.,0.,0.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 11.547005383792508));

  Dist = s14.DistanceToOut(Vec_t(20., 0., 0.), vz);
  // std::cout<<"s14.DistanceToOut(Vec_t(20.,0.,0.),vz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 11.547005383792508));

  Dist = s14.DistanceToOut(Vec_t(20., 0., 0.), vmz);
  // std::cout<<"s14.DistanceToOut(Vec_t(20.,0.,0.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 34.641016151377549));

  Dist = sb830.DistanceToOut(pb830, vb830);
  // std::cout<<"sb830.DistanceToOut(pb830,vb830... = "<<Dist<<std::endl;
  inside = sb830.Inside(pb830 + Dist * vb830);
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);
  // std::cout<<"sb830.Inside(pb830+Dist*vb830) = "<<inside<<std::endl ;

  Dist = sn11.DistanceToOut(Vec_t(0., 0., 20.), vmz);
  // std::cout<<"sn11.DistanceToOut(Vec_t(0.,0.,20.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20.));

  Dist = sn11.DistanceToOut(Vec_t(0., 0., 0.), vmz);
  // std::cout<<"sn11.DistanceToOut(Vec_t(0.,0.,0.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sn11.DistanceToOut(Vec_t(0., 0., 0.), vz);
  // std::cout<<"sn11.DistanceToOut(Vec_t(0.,0.,0.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50.));

  Dist = sn11.DistanceToOut(Vec_t(10., 0., 0.), vmz);
  // std::cout<<"sn11.DistanceToOut(Vec_t(10.,0.,0.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sn12.DistanceToOut(Vec_t(0., 0., 20.), vxmz);
  // std::cout<<"sn12.DistanceToOut(Vec_t(0.,0.,20.),vxmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20. / std::sqrt(2.)));

  Dist = sn12.DistanceToOut(Vec_t(0., 0., 0.), vmz);
  // std::cout<<"sn12.DistanceToOut(Vec_t(0.,0.,0.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sn12.DistanceToOut(Vec_t(10., 0., 10.), vz);
  // std::cout<<"sn12.DistanceToOut(Vec_t(10.,0.,10.),vz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 38.989794855663561179));

  Dist = sn12.DistanceToOut(Vec_t(10., 0., 10.), vmz);
  // std::cout<<"sn12.DistanceToOut(Vec_t(10.,0.,10.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sn14.DistanceToOut(Vec_t(10., 0., 10.), vz);
  // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,10.),vz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sn14.DistanceToOut(Vec_t(10., 0., 5.), vz);
  // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,5.),vz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 5.));

  Dist = sn14.DistanceToOut(Vec_t(10., 0., 5.), vmxz);
  // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,5.),vmxz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 3.5355339059327381968));

  Dist = sn14.DistanceToOut(Vec_t(10., 0., 10.), vmxz);
  // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,10.),vmxz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sn12.DistanceToOut(Vec_t(0., 0., 0.), vx);
  // std::cout<<"sn12.DistanceToOut(Vec_t(0.,0.,0.),vx... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  pRand = Vec_t(16.20075504802145, -22.42917454903122, -41.6469406430184);
  vRand = Vec_t(-0.280469198715188, 0.4463870649961534, 0.8497503261406731);
  valid = sn17.Normal(pRand, normal);
  // std::cout<<"norm = sn17.Normal(pRand) = "<<norm.x()<<", "<<norm.y()<<", "<<norm.z()<<std::endl;
  VECGEOM_ASSERT(valid);
  inside = sn17.Inside(pRand);
  // std::cout<<"sn17.Inside(pRand) = "<<inside<<std::endl ;
  VECGEOM_ASSERT(inside == vecgeom::EInside::kSurface);
  Dist = sn17.DistanceToOut(pRand, vRand);
  // std::cout<<"sn17.DistanceToOut(pRand,vRand,...) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 48.95871780757848768));

  // Checking Sphere_t::SafetyToIn(P)

  Dist = s2.SafetyToIn(pzero);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 45));
  Dist = s1.SafetyToIn(ponrmax1);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  // Checking Sphere_t::DistanceToIn(P,V)

  pRand = Vec_t(-10.276604989981144911, -4.7072500741022338389, 1);
  vRand = Vec_t(-0.28873162920537848164, 0.51647890054938394577, -0.80615357816219324061);
  Dist  = sn17.DistanceToIn(pRand, vRand);
  // std::cout<<"sn17.DistanceToIn(pRand,vRand) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 1.4874615083453075481));

  Dist = s1.DistanceToIn(ponzmax, vz);
  // std::cout<<"s1.DistanceToIn(ponzmax,vz) = "<<Dist<<std::endl;
  Dist = s1.DistanceToIn(pbigy, vy);
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = s1.DistanceToIn(pbigy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));

  Dist = s2.DistanceToIn(pzero, vy);
  // std::cout<<"s2.DistanceToIn(pzero,vx) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 45));
  Dist = s2.DistanceToIn(pzero, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 45));
  Dist = s2.DistanceToIn(ponrmin1, vx);
  // std::cout<<"s2.DistanceToIn(ponmin1,vx) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(Dist == 0);
  Dist = s2.DistanceToIn(ponrmin1, vmx);
  // std::cout<<"Dist of Line 872 : "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 90));

  Dist = s2.DistanceToIn(ponrmin2, vx);
  VECGEOM_ASSERT(Dist == 0);
  Dist = s2.DistanceToIn(ponrmin2, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 90 / std::sqrt(2.)));

  Dist = s3.DistanceToIn(ptesttheta1, vmz);
  //    std::cout<<"s3.DistanceToIn(ptesttheta1,vmz) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - 48 / std::sqrt(2.)));
  Dist = s3.DistanceToIn(pontheta1, vz);
  //    std::cout<<"s3.DistanceToIn(pontheta1,vz) = "<<Dist<<std::endl;
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = s3.DistanceToIn(pontheta1, vmz);
  VECGEOM_ASSERT(Dist == 0);
  Dist = s3.DistanceToIn(pontheta2, vz);
  //    std::cout<<"s3.DistanceToIn(pontheta2,vz) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(Dist == 0);
  Dist = s3.DistanceToIn(pontheta2, vmz);
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = s32.DistanceToIn(pontheta1, vz);
  //    std::cout<<"s32.DistanceToIn(pontheta1,vz) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(Dist == 0);
  Dist = s32.DistanceToIn(pontheta1, vmz);
  //    std::cout<<"s32.DistanceToIn(pontheta1,vmz) = "<<Dist<<std::endl;
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = s33.DistanceToIn(pontheta2, vz);
  //    std::cout<<"s33.DistanceToIn(pontheta2,vz) = "<<Dist<<std::endl;
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  Dist = s33.DistanceToIn(pontheta2, vmz);
  //    std::cout<<"s33.DistanceToIn(pontheta2,vmz) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(Dist == 0);

  Dist = s4.DistanceToIn(pbigy, vmy);
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  // VECGEOM_ASSERT(Dist==kInfLength);
  Dist = s4.DistanceToIn(pbigz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50));
  Dist = s4.DistanceToIn(pzero, vy);
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  // VECGEOM_ASSERT(Dist==kInfLength);
  Dist = s4.DistanceToIn(pzero, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 45));

  Dist = s4.DistanceToIn(ptestphi1, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 + 45 / std::sqrt(2.)));
  Dist = s4.DistanceToIn(ponphi1, vmxmy);
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  // VECGEOM_ASSERT(Dist==kInfLength);
  Dist = s4.DistanceToIn(ponphi1, vxy);
  //     std::cout<<"s4.DistanceToIn(ponphi1,vxy) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  Dist = s4.DistanceToIn(ptestphi2, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 + 45 / std::sqrt(2.)));
  Dist = s4.DistanceToIn(ponphi2, vmxy);
  if (Dist >= kInfLength) Dist = kInfLength;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, kInfLength));
  // VECGEOM_ASSERT(Dist==kInfLength);
  Dist = s4.DistanceToIn(ponphi2, vxmy);
  //     std::cout<<"s4.DistanceToIn(ponphi2,vxmy) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0));

  Dist = s3.DistanceToIn(pzero, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 45));
  Dist = s3.DistanceToIn(ptesttheta1, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 100 - 48 / std::sqrt(2.)));
  Dist = b216.DistanceToIn(p216, v216);
  // std::cout<<"b216.DistanceToIn(p216,v216) = "<<Dist<<std::endl;

  Dist = b658.DistanceToIn(pzero, vz);
  // std::cout<<"b658.DistanceToIn(pzero,vz) = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 209.59999999999999432));

  Dist = s13.DistanceToIn(Vec_t(20., 0., -70.), vz);
  // std::cout<<"s13.DistanceToIn(Vec_t(20.,0.,-70.),vz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 58.452994616207498));

  Dist = s13.DistanceToIn(Vec_t(20., 0., 70.), vmz);
  // std::cout<<"s13.DistanceToIn(Vec_t(20.,0.,70.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 35.358983848622451));

  Dist = s14.DistanceToIn(Vec_t(20., 0., -70.), vz);
  // std::cout<<"s14.DistanceToIn(Vec_t(20.,0.,-70.),vz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 35.358983848622451));

  Dist = s14.DistanceToIn(Vec_t(20., 0., 70.), vmz);
  // std::cout<<"s14.DistanceToIn(Vec_t(20.,0.,70.),vmz... = "<<Dist<<std::endl;
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 58.452994616207498));

  Dist = b1046.DistanceToIn(Vec_t(0., 0., 4800 * 1e6), vmz);
  std::cout << "Distance : " << Dist << std::endl;
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));
  // std::cout<<"b1046.DistanceToIn(Vec_t(0.,0.,4800*km),vmz... = "<<Dist<<std::endl;
  // if( Dist >= kInfLength ) Dist = kInfLength;
  // VECGEOM_ASSERT(ApproxEqual<Precision>(Dist,kInfLength));

  // Check DistanceToOut for  center point

  Sphere_t sntest("sntest", 0, 50, 0, 2 * PI, 0.0, 0.75 * PI);
  Vec_t in1, in2;
  in1           = Vec_t(0.0, 0.0, 30); // 0,0,30);
  in2           = Vec_t(0, 0, 0);
  double length = (in1 - in2).Mag();
  Vec_t dir12   = (in2 - in1) / length;
  // for(int i = 0 ; i < 100; i++)
  int cnt = 0;
  for (int i = 0; i < 100; i++) {
    if (sntest.Inside(in1) != vecgeom::EInside::kOutside) {
      std::cout << std::setprecision(25);

      Dist = sntest.DistanceToOut(in1, dir12);

      double diff = Dist - length;

      if (diff > 1e-6) {
        std::cout << " i=" << i << " Dout=" << Dist << " dif=" << Dist - length << std::endl;
        std::cout << "In-1 : " << in1 << "  :: Dir : " << dir12 << std::endl;
        cnt++;
      }
      in1 = in1 + Vec_t(0.00001, 0.00001, 0);
      // in2=in2+Vec_t(-0.00001,-0.00001,0);
      length = (in1 - in2).Mag();
      dir12  = (in2 - in1) / length;
    } else {
      std::cout << " error in test In=" << sntest.Inside(in1) << std::endl;
    }
  }
  std::cout << "\n--------------------------------------------------------------------\n";
  std::cout << "Number of Points with Unexpected Distance Values  : " << cnt << std::endl;
  std::cout << "--------------------------------------------------------------------\n";

  //
  // End adding unit test from Geant4
  //

  //
  // Adding some more test cases
  //

  Sphere_t sntestB("sntestB", 0, 50, 0, 2 * PI, 0.50 * PI, 0.50 * PI); // Bottom Hemisphere
  Dist = sntestB.DistanceToOut(pzero, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50.));

  Dist = sntestB.DistanceToOut(pzero, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sntestB.DistanceToOut(px, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sntestB.DistanceToOut(px, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40.));

  Dist = sntestB.DistanceToOut(py, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Dist = sntestB.DistanceToOut(py, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40.));

  Dist = sntestB.DistanceToOut(pmz, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30.));

  Dist = sntestB.DistanceToOut(pmz, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20.));

  Dist = sntestB.DistanceToOut(Vec_t(30, 0, -30), vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 30.));

  Dist = sntestB.DistanceToOut(Vec_t(30, 0, -30), vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 10.));

  Dist = sntestB.DistanceToOut(px, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20.));

  Dist = sntestB.DistanceToOut(px, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80.));

  Dist = sntestB.DistanceToOut(pmx, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80.));

  Dist = sntestB.DistanceToOut(pmx, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20.));

  Dist = sntestB.DistanceToOut(pmy, vy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 80.));

  Dist = sntestB.DistanceToOut(pmy, vmy);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 20.));

  Dist = sntestB.DistanceToOut(pmy, vx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40.));

  Dist = sntestB.DistanceToOut(pmy, vmx);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 40.));

  Dist = sntestB.DistanceToIn(pzero, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  Vec_t southPolePoint(0, 0, -50);
  Dist = sntestB.DistanceToIn(southPolePoint, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  VECGEOM_ASSERT(sntestB.Inside(southPolePoint) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(sntestB.Inside(pzero) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(sntestB.Inside(px) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(sntestB.Inside(-py) == vecgeom::EInside::kSurface);

  Vec_t northPolePoint(0, 0, 50);
  Dist = sntestB.DistanceToIn(northPolePoint, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50.));

  VECGEOM_ASSERT(sntestB.Inside(northPolePoint) == vecgeom::EInside::kOutside);

  Dist = sntest.DistanceToIn(southPolePoint, vz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 50.));
  VECGEOM_ASSERT(sntest.Inside(southPolePoint) == vecgeom::EInside::kOutside);

  Dist = sntest.DistanceToIn(northPolePoint, vmz);
  VECGEOM_ASSERT(ApproxEqual<Precision>(Dist, 0.));

  VECGEOM_ASSERT(sntest.Inside(northPolePoint) == vecgeom::EInside::kSurface);

  VECGEOM_ASSERT(sntest.Inside(southPolePoint) == vecgeom::EInside::kOutside);

  VECGEOM_ASSERT(sntest.Inside(px) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(sntest.Inside(py) == vecgeom::EInside::kInside);
  // std::cout<<"Location of Px : "<<sntest.Inside(px)<<std::endl;

  VECGEOM_ASSERT(sntest.Inside(pz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(sntest.Inside(-pz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(sntestB.Inside(pz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(sntestB.Inside(-pz) == vecgeom::EInside::kInside);

  // Added a new test for normal
  Sphere_t sphNormal("normalTest", 80, 100, 0, 2 * PI, 0, 2.268928027592628);
  Vec_t ptNORMAL(-57.1682, -73.6533, -32.3008);
  VECGEOM_ASSERT(sphNormal.Inside(ptNORMAL) == vecgeom::EInside::kInside);
  valid = sphNormal.Normal(ptNORMAL, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(normal.Mag(), 1.));
  VECGEOM_ASSERT(!valid); // because the point under test is Inside Point

  // added more precise point (reported by Gabriele), which makes it a surface point
  Vec_t ptNormalSurface(-58.16821296689909, -74.65330731271325, -32.30081589545529);
  VECGEOM_ASSERT(sphNormal.Inside(ptNormalSurface) == vecgeom::EInside::kSurface);
  valid = sphNormal.Normal(ptNormalSurface, normal);
  VECGEOM_ASSERT(ApproxEqual<Precision>(normal.Mag(), 1.));
  VECGEOM_ASSERT(valid); // because the point under test is now OnSurface

  Vec_t ptOnEndThetaSurface(46.76611644842609, -40.66413203238494, -52.00144394546794); // Point on the EndTheta surface
  valid = sphNormal.Normal(ptOnEndThetaSurface, normal);
  VECGEOM_ASSERT(sphNormal.Inside(ptOnEndThetaSurface) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(ApproxEqual<Precision>(normal.Mag(), 1.));
  VECGEOM_ASSERT(valid); // because the point under test is on End Theta Surface

  return true;
}

int main(int argc, char *argv[])
{
  VECGEOM_ASSERT(TestSphere<vecgeom::SimpleSphere>());
  std::cout << "VecGeomSphere passed\n";

  return 0;
}
