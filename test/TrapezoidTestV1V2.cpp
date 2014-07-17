/*
 * @file   test/TrapezoidTest.cpp
 * @author Guilherme Lima (lima at fnal dot gov)
 *
 * 140516 G.Lima - created from Johannes' parallelepiped example
 */
#include <iostream>
#include "volumes/UnplacedVolume.h"
#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedTrapezoid.h"
#include "volumes/SpecializedTrapezoid.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "volumes/kernel/TrapezoidImplementation.h"
#include "navigation/SimpleNavigator.h"
#include "TGeoArb8.h"

#include "VUSolid.hh"
#include "UTrap.hh"
#include "base/RNG.h"

using namespace VECGEOM_NAMESPACE;

typedef kScalar Backend;

// void compile_test() {
//  UnplacedTrapezoid trap(10, 20, 30, 0, M_PI/4);
//  Vector3D<typename Backend::precision_v> point(1.0, 1.5, 2.5);
//  Vector3D<typename Backend::precision_v> dir(1.0, 1.5, 2.5);

//  TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);
//  typename Backend::bool_v inside;
//  typename Backend::precision_v distance;
// }

int debug = 1;

TGeoTrap convertToRoot(UnplacedTrapezoid const& trap) {
  return TGeoTrap( trap.GetDz(), trap.GetTheta()*kRadToDeg, trap.GetPhi()*kRadToDeg,
                   trap.GetDy1(), trap.GetDx1(), trap.GetDx2(), trap.GetTanAlpha1(),
                   trap.GetDy2(), trap.GetDx3(), trap.GetDx4(), trap.GetTanAlpha2() );
}

UTrap convertToUSolids(UnplacedTrapezoid const& trap) {
  return UTrap( "trap", trap.GetDz(), trap.GetTheta(), trap.GetPhi(),
                   trap.GetDy1(), trap.GetDx1(), trap.GetDx2(), trap.GetTanAlpha1(),
                   trap.GetDy2(), trap.GetDx3(), trap.GetDx4(), trap.GetTanAlpha2() );
}

bool testVolume(UnplacedTrapezoid const& trap, TGeoTrap const& rtrap) {
  auto vecgVol = trap.Volume();
  auto rootVol = rtrap.Capacity();
  if(debug>2) printf("Volume test: Root: %f  --  Vecgeom: %f\n", rootVol, vecgVol );

  bool good = fabs( vecgVol-rootVol ) < kTolerance;
  if(good && debug) printf("testVolume vs. Root passed.\n");

  return good;
}

bool testVolume(UnplacedTrapezoid const& trap, UTrap& utrap) {
  auto vecgVol = trap.Volume();
  auto usolVol = utrap.Capacity();
  if(debug>0) printf("Volume test: USolids: %f  --  Vecgeom: %f\n", usolVol, vecgVol );

  bool good = fabs( vecgVol-usolVol ) < kTolerance;
  if(good && debug) printf("testVolume vs. USolids passed.\n");
  else printf("*** testVolume vs. USolids FAILED!   USolids: %f  --  Vecgeom: %f\n", usolVol, vecgVol );;

  return good;
}

bool testCorners(UnplacedTrapezoid const& trap, TGeoTrap& rtrap) {

  // get corners from vecgeom trapezoid
  TrapCorners_t pt;
  trap.fromParametersToCorners(pt);

  // get corners from root trapezoid
  Double_t const* vtx = rtrap.GetVertices();

  // order of points is different between root and Usolids/vecgeom!!
  bool good = true;
  unsigned int iroot[8] = {0,3,1,2,4,7,5,6};
  for(int i=0; i<8; ++i) {
    auto ir = iroot[i];
    Precision zr = ( ir>3 ? rtrap.GetDz() : -rtrap.GetDz() );

    if(debug>1) {
      printf("Corner %d: {%.2f, %.2f, %.2f}\t\t{%.2f, %.2f, %.2f}\n", i,
             pt[i][0], pt[i][1], pt[i][2], vtx[2*ir+0], vtx[2*ir+1], zr);
    }

    if( fabs( pt[i][0] - vtx[2*ir+0] ) > kTolerance ) {
      if (debug>0) printf("Discrepancy in corner x-coords: %.2f vs. %.2f\n", pt[i][0], vtx[2*ir+0]);
      good = false;
    }

    if( fabs( pt[i][1] - vtx[2*ir+1] ) > kTolerance ) {
      if(debug>0) printf("Discrepancy in corner y-coords: %.2f vs. %.2f\n", pt[i][1], vtx[2*ir+1]);
      good = false;
    }

    if( fabs(pt[i][2] - zr ) > kTolerance ) {
      if(debug>0) printf("Discrepancy in corner z-coords: %.2f vs. %.2f\n", pt[i][2], zr);
      good = false;
    }
  }

  if(good && debug) printf("testCorners vs. ROOT passed.\n");
  return good;
}


bool testPlanes(UnplacedTrapezoid const& trap, UTrap& utrap) {

  // get planes from vecgeom trapezoid
  // TrapSidePlane const* planes = trap.GetPlanes();
  Planes const* planes = trap.GetPlanes2();

  // get planes from usolids trapezoid
  UTrapSidePlane uplanes[4];

  // order of points is different between root and Usolids/vecgeom!!
  bool good = true;
  // unsigned int iusolids[8] = {0,1,2,3,4,5,6,7};
  for(int i=0; i<4; ++i) {
    uplanes[i] = utrap.GetSidePlane(i);
    // auto ir = iusolids[i];
    // Precision zr = ( ir>3 ? utrap.GetZHalfLength() : -utrap.GetZHalfLength() );

    if(debug>2) {
      printf("Plane %d: {%.3f, %.3f, %.3f, %.3f}\t\t{%.3f, %.3f, %.3f, %.3f}\n", i,
             planes->fA[i], planes->fB[i], planes->fC[i], planes->fD[i], uplanes[i].a, uplanes[i].b, uplanes[i].c, uplanes[i].d);
    }

    if( fabs( planes->fA[i] - uplanes[i].a ) > kTolerance ) { // V2
      printf("Discrepancy in plane A-value: %.3f vs. %.3f\n", planes->fA[i], uplanes[i].a);
      good = false;
    }

    if( fabs( planes->fB[i] - uplanes[i].b ) > kTolerance ) { // V2
      printf("Discrepancy in plane B-value: %.3f vs. %.3f\n", planes->fB[i], uplanes[i].b);
      good = false;
    }

    if( fabs( planes->fC[i] - uplanes[i].c ) > kTolerance ) { // V2
      printf("Discrepancy in plane C-value: %.3f vs. %.3f\n", planes->fC[i], uplanes[i].c);
      good = false;
    }

    if( fabs( planes->fD[i] - uplanes[i].d ) > kTolerance ) { // V2
      printf("Discrepancy in plane D-value: %.3f vs. %.3f\n", planes->fD[i], uplanes[i].d);
      good = false;
    }
  }

  if(good && debug) printf("testPlanes vs. USolids passed.\n");
  return good;
}


void insideRoot(PlacedTrapezoid const& trap, TGeoTrap const& rtrap) {

  bool passed = true;

  typename Backend::bool_v inside_v;
  bool inside;
  double p[3];

  int in = 0, out = 0;
  for(int i = 0; i < 10000; i++) {
    double x = RNG::Instance().uniform(-10,10);
    double y = RNG::Instance().uniform(-10,10);
    double z = RNG::Instance().uniform(-20,20);

    p[0] = x;
    p[1] = y;
    p[2] = z;

    Vector3D<typename Backend::precision_v> point(x, y, z);

    inside_v = trap.Inside(point);
    inside = rtrap.Contains(p);

    if(inside)
      in++;
    else
      out++;

    if(inside != inside_v) {
      std::cout << "ERROR for point " << point << ": " << inside <<' '<< inside_v << std::endl;
      passed = false;
    }
  }

  std::cout << "insideRoot(): in: " << in << " out: " << out << std::endl;
  if(passed) printf("insideRoot() test passed.\n");
}

void insideUSolids(PlacedTrapezoid const& trap, UTrap const& utrap) {

  bool passed = true;

  int in = 0, out = 0;
  for(int i = 0; i < 10000; i++) {
    double x = RNG::Instance().uniform(-10,10);
    double y = RNG::Instance().uniform(-10,10);
    double z = RNG::Instance().uniform(-20,20);

    Vector3D<typename Backend::precision_v> point(x, y, z);

    typename Backend::bool_v vginside = trap.Inside(point);
    VUSolid::EnumInside uinside = utrap.Inside(point);

    if(vginside)      in++;
    else      out++;

    bool good = false;
    // if( vginside==EInside::kInside  && uinside==::VUSolid::eInside  ) good = true;
    // if( vginside==EInside::kOutside && uinside==::VUSolid::eOutside ) good = true;
    // if( vginside==EInside::kSurface && uinside==::VUSolid::eSurface ) good = true;
    if( vginside == uinside ) good = true;
    if(!good) {
      std::cout << "ERROR for point " << point << ": "
                << vginside <<" "<< (vginside==EInside::kInside?"(in) / ":"(out) / ")
                << uinside <<" "<< (uinside ==EInside::kInside?"(in)":"(out)")
                << std::endl;
      passed = false;
    }
  }

  std::cout << "insideUSolids: in: " << in << " out: " << out << std::endl;
  if(passed) printf("insideUSolids() test passed.\n");
}

void distancetoin(PlacedTrapezoid const& trap, TGeoTrap const& rtrap) {

  typename Backend::precision_v vgdist = kInfinity;
  Precision rdist;
  double p[3], v[3];

  int in = 0, misses = 0, hits = 0;
  for(int i = 0; i < 100; i++) {
    // double x = RNG::Instance().uniform(-7,7);
    // double y = RNG::Instance().uniform(-7,7);
    // double z = RNG::Instance().uniform(-7,7);
    double x = -10. + 0.2*float(i);
    double y = 0;
    double z = 0;

    p[0] = x;
    p[1] = y;
    p[2] = z;

    Vector3D<typename Backend::precision_v> point(x, y, z);
    // Vector3D<typename Backend::precision_v> direction = volumeUtilities::SampleDirection();
    Vector3D<typename Backend::precision_v> direction(1,0,0);

    v[0] = direction.x();
    v[1] = direction.y();
    v[2] = direction.z();

    Vector3D<Precision> vpos(p[0],p[1],p[2]);
    Vector3D<Precision> vdir(v[0],v[1],v[2]);
    vgdist = trap.DistanceToIn( vpos, vdir );

    rdist  = rtrap.DistFromOutside(p, v);

    if(  rdist == 1e+30 && vgdist == kInfinity ) {
      std::cout << "OK: Inside="<< rtrap.Contains(p) << " - dist for point " << vpos << " w dir " << vdir << ": " << vgdist << " , " << rdist << std::endl;
      ++hits;
    }
    else {
      if( fabs(rdist-vgdist) < kTolerance ) {
      std::cout << "OK: Inside="<< rtrap.Contains(p) << " - dist for point " << vpos << " w dir " << vdir << ": " << vgdist << " , " << rdist << std::endl;
        ++hits;
      }
      else {
        ++misses;
        std::cout << "ERROR: Inside="<< rtrap.Contains(p) << " - dist for point " << vpos << " w dir " << vdir << ": " << vgdist << " , " << rdist << std::endl;
      }
    }
  }
  std::cout << "distancetoin: in: " << in << " misses: " << misses << " hits: " << hits << std::endl;
}

void safety() {
  UnplacedTrapezoid trap(5,0,0,5,5,5,0,5,5,5,0);
  TGeoTrap rtrap(5,0,0,5,5,5,0,5,5,5,0);
  // TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

  typename Backend::precision_v safety_v = kInfinity;
  double saf;
  double p[3];

  for(int i = 0; i < 1000; i++) {
    double x = RNG::Instance().uniform(-30,30);
    double y = RNG::Instance().uniform(-30,30);
    double z = RNG::Instance().uniform(-40,40);

    p[0] = x;
    p[1] = y;
    p[2] = z;

    Vector3D<typename Backend::precision_v> point(x, y, z);
    saf = rtrap.Safety(p, false);

    if(!rtrap.Contains(p)) {

      if(Abs(saf-safety_v) > 0.0001) {
        std::cout << "ERROR for point " << point << "\t" << saf << std::endl;
      }
    }

    // if(inside != inside_v)
    //   std::cout << "ERROR for point " << point << ": " << inside << inside_v << std::endl;
  }

  // std::cout << "inside: in: " << in << " out: " << out << std::endl;
}


int main() {

  UnplacedTrapezoid world_params = UnplacedTrapezoid(4.,0,0, 4.,4.,4.,0, 4.,4.,4.,0);
  UnplacedTrapezoid largebox_params = UnplacedTrapezoid(1.5, 0,0, 1.5,1.5,1.5,0, 1.5,1.5,1.5,0 );

  LogicalVolume worldl = LogicalVolume(&world_params);
  LogicalVolume largebox = LogicalVolume("Trapezoid box", &largebox_params);

  // Transformation3D origin = Transformation3D();
  Transformation3D placement1 = Transformation3D( 2,  2,  2);
  Transformation3D placement2 = Transformation3D(-2,  2,  2);
  // Transformation3D placement3 = Transformation3D( 2, -2,  2);
  // Transformation3D placement4 = Transformation3D( 2,  2, -2);
  // Transformation3D placement5 = Transformation3D(-2, -2,  2);
  // Transformation3D placement6 = Transformation3D(-2,  2, -2);
  // Transformation3D placement7 = Transformation3D( 2, -2, -2);
  // Transformation3D placement8 = Transformation3D(-2, -2, -2);

  // largebox.PlaceDaughter(&smallbox, &origin);
  worldl.PlaceDaughter(&largebox, &placement1);
  worldl.PlaceDaughter(&largebox, &placement2);
  // worldl.PlaceDaughter(&largebox, &placement3);
  // worldl.PlaceDaughter(&largebox, &placement4);
  // worldl.PlaceDaughter("Hello the world!", &largebox, &placement5);
  // worldl.PlaceDaughter(&largebox, &placement6);
  // worldl.PlaceDaughter(&largebox, &placement7);
  // worldl.PlaceDaughter(&largebox, &placement8);

  VPlacedVolume *world_placed = worldl.Place();

  std::cerr << "Printing world content:\n";
  world_placed->PrintContent();

  std::cerr<<"\nInstantiating a SimpleNavigator...\n";
  SimpleNavigator nav;
  Vector3D<Precision> point(2, 2, 2);
  NavigationState path(4);
  std::cerr<<"SimpleNavigator test...\n";
  nav.LocatePoint(world_placed, point, path, true);
  std::cerr<<"\nSimpleNavigator.LocatePoint result...\n";
  path.Print();

  std::cerr<<"\nFindLogicalVolume(LargeBox)...\n";
  GeoManager::Instance().FindLogicalVolume("Trapezoid box");
  GeoManager::Instance().FindPlacedVolume("Trapezoid box");

//=======  VALIDATION SECTION - comparing with equivalent USOLIDS shape

  UnplacedTrapezoid trap;
  TGeoTrap rtrap;

  // validate basic constructor
//  trap = UnplacedTrapezoid(5,0,0,5,2,3,0,10,4,6,0);
  trap = UnplacedTrapezoid(5,0,0,5,5,5,0,5,5,5,0);
  rtrap = convertToRoot(trap);
  UTrap utrap = convertToUSolids(trap);
  if(debug>0) {
    trap.Print();
    printf("In degrees: Theta=%.2f;  Phi=%.2f\n", trap.GetTheta()*kRadToDeg, trap.GetPhi()*kRadToDeg);
  }
  testVolume(trap, rtrap);
  testVolume(trap, utrap);
  testCorners(trap, rtrap);
  testPlanes(trap, utrap);

  // validate constructor with input array
  double pars[11] = {15,0,0,5,5,15,0,10,25,45,0};
  trap = UnplacedTrapezoid( pars );
  rtrap = convertToRoot(trap);
  UTrap utrap2 = convertToUSolids(trap);
  if(debug>0) {
    trap.Print();
    printf("In degrees: Theta=%.2f;  Phi=%.2f\n", trap.GetTheta()*kRadToDeg, trap.GetPhi()*kRadToDeg);
  }
  testVolume(trap, rtrap);
  testCorners(trap, rtrap);
  testVolume(trap, utrap2);
  testPlanes(trap, utrap2);

  // validate construtor for input corner points -- add an xy-offset for non-zero theta,phi
  TrapCorners_t xyz;
  Precision xoffset = 9;
  Precision yoffset = -6;
  // convention: p0(---); p1(+--); p2(-+-); p3(++-); p4(--+); p5(+-+); p6(-++); p7(+++)
  xyz[0] = Vector3D<Precision>( -2+xoffset, -5+yoffset, -15 );
  xyz[1] = Vector3D<Precision>(  2+xoffset, -5+yoffset, -15 );
  xyz[2] = Vector3D<Precision>( -3+xoffset,  5+yoffset, -15 );
  xyz[3] = Vector3D<Precision>(  3+xoffset,  5+yoffset, -15 );
  xyz[4] = Vector3D<Precision>( -4-xoffset,-10-yoffset,  15 );
  xyz[5] = Vector3D<Precision>(  4-xoffset,-10-yoffset,  15 );
  xyz[6] = Vector3D<Precision>( -6-xoffset, 10-yoffset,  15 );
  xyz[7] = Vector3D<Precision>(  6-xoffset, 10-yoffset,  15 );

  trap = UnplacedTrapezoid(xyz);
  rtrap = convertToRoot(trap);
  UTrap utrap3 = convertToUSolids(trap);
  if(debug>0) {
    trap.Print();
    printf("In degrees: Theta=%.2f;  Phi=%.2f\n", trap.GetTheta()*kRadToDeg, trap.GetPhi()*kRadToDeg);
  }
  testVolume(trap, rtrap);
  testVolume(trap, utrap3);
  testCorners(trap, rtrap);
  testPlanes(trap, utrap3);

  PlacedTrapezoid const* ptrap = reinterpret_cast<PlacedTrapezoid const*>( LogicalVolume(&trap).Place() );
  insideRoot(*ptrap,rtrap);
  insideUSolids(*ptrap,utrap3);

  distancetoin(*ptrap, rtrap);
  // safety();
}
