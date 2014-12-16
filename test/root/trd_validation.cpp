#include <iostream>
#include "volumes/UnplacedTrd.h"
#include "volumes/kernel/TrdImplementation.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "TGeoTrd2.h"
#include "base/RNG.h"

using namespace VECGEOM_NAMESPACE;

typedef kScalar Backend;

// void compile_test() {
//  UnplacedTube tube(10, 20, 30, 0, M_PI/4);
//  Vector3D<typename Backend::precision_v> point(1.0, 1.5, 2.5);
//  Vector3D<typename Backend::precision_v> dir(1.0, 1.5, 2.5);

//  TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);
//  typename Backend::bool_v inside;
//  typename Backend::precision_v distance;
//  TubeUnplacedInside<Backend, TubeTraits::NonHollowTubeWithAcuteSector>(tube, point, &inside);
//  TubeDistanceToIn<0, 1, Backend, TubeTraits::NonHollowTubeWithAcuteSector>(tube, *identity, point, dir, kInfinity, &distance);

// }

double x1 = 5.;
double x2 = 10.;
double ay1 = 4.;
//double ay1 = 9.;
double y2 = 4.;
double z = 10;

#define NPOINTS 1000000

void inside() {
  UnplacedTrd trd(x1, x2, ay1, y2, z);
  TGeoTrd2 rtrd(x1, x2, ay1, y2, z);

  typename Backend::bool_v inside_v;
  bool inside;
  double p[3];

  int in = 0, out = 0;
  for(int i = 0; i < NPOINTS; i++) {
    double x = RNG::Instance().uniform(-30,30);
    double y = RNG::Instance().uniform(-30,30);
    double z = RNG::Instance().uniform(-40,40);

    p[0] = x;
    p[1] = y;
    p[2] = z;

    Vector3D<typename Backend::precision_v> point(x, y, z);
    inside = rtrd.Contains(p);

    TrdImplementation<rotation::kIdentity, translation::kIdentity, TrdTypes::UniversalTrd> impl;
    impl.UnplacedContains<Backend>(trd, point, inside_v);
    
    if(inside)
      in++;
    else
      out++;

    if(inside != inside_v)
      std::cout << "ERROR for point " << point << ": " << inside << inside_v << std::endl;

  }

  std::cout << "inside: in: " << in << " out: " << out << std::endl;
}

void distancetoin() {
  UnplacedTrd trd(x1, x2, ay1, y2, z);
  TGeoTrd2 rtrd(x1, x2, ay1, y2, z);
  Transformation3D const * identity = new Transformation3D(0,0,0,0,0,0);
  typename Backend::precision_v dist_v;
  typename Backend::precision_v dist;
  double p[3], v[3];

  int in = 0, misses = 0, hits = 0, correct = 0, errors = 0;
  for(int i = 0; i < NPOINTS; i++) {
    double x = RNG::Instance().uniform(-30,30);
    double y = RNG::Instance().uniform(-30,30);
    double z = RNG::Instance().uniform(-40,40);

    p[0] = x;
    p[1] = y;
    p[2] = z;

    Vector3D<typename Backend::precision_v> point(x, y, z);
    Vector3D<typename Backend::precision_v> direction = volumeUtilities::SampleDirection();

    v[0] = direction.x();
    v[1] = direction.y();
    v[2] = direction.z();

#ifndef VECGEOM_NO_SPECIALIZATION
    TrdImplementation<rotation::kIdentity, translation::kIdentity, TrdTypes::Trd1> impl;
#else
    TrdImplementation<rotation::kIdentity, translation::kIdentity, TrdTypes::UniversalTrd> impl;
#endif
    impl.DistanceToIn<Backend>(trd, *identity, point, direction, kInfinity, dist_v);
    dist = rtrd.DistFromOutside(p, v);

    if(!rtrd.Contains(p)) {
      if(dist == 1e+30)
        misses++;
      else
        hits++;

      if(  !(dist == 1e+30 && dist_v == kInfinity) ) {

        if( Abs(dist-dist_v) >= kTolerance ) {
          std::cout << "ERROR: dist for point " << point << " w dir " << direction << ": " << dist << " , " << dist_v << std::endl;
          errors++;
        }
        else
          correct++;
      }
    }
    else {
      in++;
    }

    // if(inside != inside_v)
    //   std::cout << "ERROR for point " << point << ": " << inside << inside_v << std::endl;
  }
  std::cout << "distancetoin: in: " << in << " misses: " << misses << " hits: " << hits << std::endl;
  std::cout << "correct: " << correct << ", errors: " << errors << std::endl;
}

void distancetoout() {
  UnplacedTrd trd(x1, x2, ay1, y2, z);
  TGeoTrd2 rtrd(x1, x2, ay1, y2, z);
  typename Backend::precision_v dist_v;
  Precision dist;
  double p[3], v[3];

  int in = 0, out = 0, correct = 0, errors = 0;
  double x, y, z;
  for(int i = 0; i < NPOINTS; i++) {
    
    do {
      x = RNG::Instance().uniform(-30,30);
      y = RNG::Instance().uniform(-30,30);
      z = RNG::Instance().uniform(-40,40);

      p[0] = x;
      p[1] = y;
      p[2] = z;
    } while(!rtrd.Contains(p));

    Vector3D<typename Backend::precision_v> point(x, y, z);
    Vector3D<typename Backend::precision_v> direction = volumeUtilities::SampleDirection();

    v[0] = direction.x();
    v[1] = direction.y();
    v[2] = direction.z();

    TrdImplementation<rotation::kIdentity, translation::kIdentity, TrdTypes::UniversalTrd> impl;
    impl.DistanceToOut<Backend>(trd, point, direction, kInfinity, dist_v);
    dist = rtrd.DistFromInside(p, v);

    if(rtrd.Contains(p)) {
        in++;
        if( Abs(dist-dist_v) >= kTolerance ) {
          std::cout << "ERROR: dist to out for point " << point << " w dir " << direction << ": " << dist << " , " << dist_v << std::endl;
          errors++;
        }
        else {
          //std::cout << "OK: dist to out for point " << point << " w dir " << direction << ": " << dist << " , " << dist_v << std::endl;
          correct++;
        }

    }
    else {
      out++;
    }
  }
  std::cout << "distancetoout: out: " << out << " in: " << in << std::endl;
  std::cout << "correct: " << correct << ", errors: " << errors << std::endl;
}


void safety() {
  UnplacedTrd trd(x1, x2, ay1, y2, z);
  TGeoTrd2 rtrd(x1, x2, ay1, y2, z);
  Transformation3D const * identity = new Transformation3D(0,0,0,0,0,0);

  typename Backend::precision_v safety_v;
  double saf;
  double p[3];
  int correct = 0, errors = 0;

  for(int i = 0; i < NPOINTS; i++) {
    double x = RNG::Instance().uniform(-30,30);
    double y = RNG::Instance().uniform(-30,30);
    double z = RNG::Instance().uniform(-40,40);

    p[0] = x;
    p[1] = y;
    p[2] = z;

    Vector3D<typename Backend::precision_v> point(x, y, z);
    saf = rtrd.Safety(p, false);

    TrdImplementation<rotation::kIdentity, translation::kIdentity, TrdTypes::UniversalTrd> impl;
    impl.SafetyToIn<Backend>(trd, *identity, point, safety_v);

    if(!rtrd.Contains(p)) {

      if(Abs(saf-safety_v) > 0.0001) {
        std::cout << "ERROR for point " << point << "\t" << saf << "\t" << safety_v << std::endl;
        errors++;
      }
      else {
        correct++;
      }
    }
  }
  std::cout << "correct: " << correct << ", errors: " << errors << std::endl;
}

void safetyout() {
  UnplacedTrd trd(x1, x2, ay1, y2, z);
  TGeoTrd2 rtrd(x1, x2, ay1, y2, z);

  typename Backend::precision_v safety_v;
  double saf;
  double p[3];
  double x, y, z;
  int correct = 0, errors = 0;

  for(int i = 0; i < NPOINTS; i++) {
    
    do {
      x = RNG::Instance().uniform(-30,30);
      y = RNG::Instance().uniform(-30,30);
      z = RNG::Instance().uniform(-40,40);

      p[0] = x;
      p[1] = y;
      p[2] = z;
    } while(!rtrd.Contains(p));

    Vector3D<typename Backend::precision_v> point(x, y, z);
    saf = rtrd.Safety(p, true);

    TrdImplementation<rotation::kIdentity, translation::kIdentity, TrdTypes::UniversalTrd> impl;
    impl.SafetyToOut<Backend>(trd, point, safety_v);

    if(rtrd.Contains(p)) {

      if(Abs(saf-safety_v) > 0.0001) {
        std::cout << "ERROR for point " << point << "\t" << saf << " " << safety_v << std::endl;
        errors++;
      }
      else {
        //std::cout << "OK for point " << point << "\t" << saf << " " << safety_v << std::endl;
        correct++;
      }
    }
  }
  std::cout << "correct: " << correct << ", errors: " << errors << std::endl;
}



int main() {
  inside();
  distancetoin();
  safety();
  distancetoout();
  safetyout();
}

