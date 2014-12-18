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

using std::max;

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

int debug = 2;

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

bool testNormal(UnplacedTrapezoid const& trap, UTrap const& utrap ) {

  bool passed = true;
  auto dx = max( max(trap.GetDx1(), trap.GetDx2()), max(trap.GetDx3(), trap.GetDx4()));
  auto dy = max(trap.GetDy1(), trap.GetDy2());
  auto dz = trap.GetDz();
  dx *= 2.0;
  dy *= 2.0;
  dz *= 2.0;

  for(int i = 0; i < 100000; i++) {
    double x = RNG::Instance().uniform(0., dx);
    double y = RNG::Instance().uniform(0., dy);
    double z = RNG::Instance().uniform(0., dz);
    Vector3D<Precision> point(x, y, z);
    if(i%2==0) {
      point = trap.GetPointOnSurface();
    }

    Vector3D<Precision> vecgNormal;
    trap.Normal(point, vecgNormal);
    UVector3 usolNormal;
    utrap.Normal(point, usolNormal);

    bool good = true;
    if( fabs(vecgNormal[0] - usolNormal.x()) > kTolerance ) good = false;
    if( fabs(vecgNormal[1] - usolNormal.y()) > kTolerance ) good = false;
    if( fabs(vecgNormal[2] - usolNormal.z()) > kTolerance ) good = false;

    if(!good) {
      passed = false;
      printf("*** testNormal vs. USolids FAILED for point: (%f; %f; %f)\n\tUSolids: (%f; %f; %f)  --  Vecgeom: (%f; %f; %f)\n",
             point.x(), point.y(), point.z(),
             usolNormal.x(), usolNormal.y(), usolNormal.z(),
             vecgNormal[0], vecgNormal[1], vecgNormal[2]);
    }
  }

  if(passed && debug) printf("testNormal: VecGeom vs. USolids passed.\n");
  return passed;
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
#ifndef VECGEOM_PLANESHELL_DISABLE
  Planes const* planes = trap.GetPlanes();
#else
  TrapSidePlane const* planes = trap.GetPlanes();
#endif

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
#ifndef VECGEOM_PLANESHELL_DISABLE
             planes->fA[i], planes->fB[i], planes->fC[i], planes->fD[i], uplanes[i].a, uplanes[i].b, uplanes[i].c, uplanes[i].d);
#else
             planes[i].fA, planes[i].fB, planes[i].fC, planes[i].fD, uplanes[i].a, uplanes[i].b, uplanes[i].c, uplanes[i].d);
#endif
    }

#ifndef VECGEOM_PLANESHELL_DISABLE
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
#else
    if( fabs( planes[i].fA - uplanes[i].a ) > kTolerance ) { // V1
      printf("Discrepancy in plane A-value: %.3f vs. %.3f\n", planes[i].fA, uplanes[i].a);
      good = false;
    }

    if( fabs( planes[i].fB - uplanes[i].b ) > kTolerance ) { // V1
      printf("Discrepancy in plane B-value: %.3f vs. %.3f\n", planes[i].fB, uplanes[i].b);
      good = false;
    }

    if( fabs( planes[i].fC - uplanes[i].c ) > kTolerance ) { // V1
      printf("Discrepancy in plane C-value: %.3f vs. %.3f\n", planes[i].fC, uplanes[i].c);
      good = false;
    }

    if( fabs( planes[i].fD - uplanes[i].d ) > kTolerance ) { // V1
      printf("Discrepancy in plane D-value: %.3f vs. %.3f\n", planes[i].fD, uplanes[i].d);
      good = false;
    }
#endif
  }

  if(good && debug) printf("testPlanes vs. USolids passed.\n");
  return good;
}


void containsRoot(PlacedTrapezoid const& trap, TGeoTrap const& rtrap) {

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

    inside_v = trap.Contains(point);
    inside = rtrap.Contains(p);

    if(inside)  in++;
    else        out++;

    if(inside != inside_v) {
      std::cout << "ERROR (containsRoot) for point " << point << ": " << inside <<' '<< inside_v << std::endl;
      passed = false;
    }
  }

  std::cout << "containsRoot(): in: " << in << " out: " << out << std::endl;
  if(passed) printf("containsRoot() test passed.\n");
}

void insideUSolids(PlacedTrapezoid const& trap, UTrap const& utrap) {

  bool passed = true;

  int in = 0, out = 0, surface=0;
  for(int i = 0; i < 10000; i++) {
    double x = RNG::Instance().uniform(-10,10);
    double y = RNG::Instance().uniform(-10,10);
    double z = RNG::Instance().uniform(-20,20);

    Vector3D<typename Backend::precision_v> point(x, y, z);

    typename vecgeom::Inside_t vginside = trap.Inside(point);
    typename vecgeom::Inside_t uinside  = utrap.Inside(point);

    if(vginside==EInside::kInside)       in++;
    else if(vginside==EInside::kSurface) surface++;
    else if(vginside==EInside::kOutside) out++;
    else std::cout<<"ERROR (insideUSolids) for point "<< point <<": "
                  <<"*** Invalid Inside_t value returned: "<< vginside <<' '<< uinside <<"\n";

    bool good = false;
    // if( vginside==EInside::kInside  && uinside==::VUSolid::eInside  ) good = true;
    // if( vginside==EInside::kOutside && uinside==::VUSolid::eOutside ) good = true;
    // if( vginside==EInside::kSurface && uinside==::VUSolid::eSurface ) good = true;
    if( vginside == uinside ) good = true;
    if(!good) {
      std::cout << "ERROR (insideUsolids) for point " << point << ": "
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
    double x = -50 + 1.*float(i);
    double y = 13;
    double z = 27;

    p[0] = x;
    p[1] = y;
    p[2] = z;

    Vector3D<typename Backend::precision_v> point(x, y, z);
    // Vector3D<typename Backend::precision_v> direction = volumeUtilities::SampleDirection();
    Vector3D<typename Backend::precision_v> direction(-0.96, -0.166, -0.22);

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
        std::cout << "ERROR (distanceToIn): Inside="<< rtrap.Contains(p) << " - dist for point " << vpos << " w dir " << vdir << ": VG=" << vgdist << ", Root=" << rdist << std::endl;
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

  UnplacedTrapezoid world_params = UnplacedTrapezoid(50.,0,0, 50.,50.,50.,0, 50.,50.,50.,0);
  UnplacedTrapezoid largebox_params = UnplacedTrapezoid(15., 0,0, 5.,2.,3.,0, 10.,4.,6.,0 );

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
  // worldl.PlaceDaughter(&largebox, &placement1);
  // worldl.PlaceDaughter(&largebox, &placement2);
  worldl.PlaceDaughter(&largebox, &Transformation3D::kIdentity);

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
  testNormal( trap, utrap );

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
  testVolume(trap, utrap2);
  // testNormal( trap, utrap2 );
  testCorners(trap, rtrap);
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
  // testNormal(trap, utrap3);
  testCorners(trap, rtrap);
  testPlanes(trap, utrap3);

  PlacedTrapezoid const* ptrap = reinterpret_cast<PlacedTrapezoid const*>( LogicalVolume(&trap).Place() );
  containsRoot(*ptrap,rtrap);
  insideUSolids(*ptrap,utrap3);
  distancetoin(*ptrap, rtrap);  // some discrepancies...

  // safety();
}
