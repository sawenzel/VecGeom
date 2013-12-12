#ifndef SHAPE_TESTER_H
#define SHAPE_TESTER_H

#include <vector>
#include <iostream>
#include <iomanip>
#include "GeoManager.h"

typedef struct {
  double fastgeom;
  double usolids;
  double root;
  void Print() const {
    std::cout << "Benchmark:\n"
              << std::setw(2) << fastgeom
              << std::endl;
  }
} ShapeBenchmark;


typedef struct {
  PhysicalVolume const *fastgeom;
  VUSolid const *usolids;
  TGeoShape const *root;
} VolumePointers;

class ShapeTester {

private:

  PhysicalVolume const *world;
  std::vector<VolumePointers> volumes;
  unsigned n_points = 1<<10;
  unsigned reps = 1e3;
  double bias = 0.8;
  std::vector<ShapeBenchmark> results;

public:

  void Run();
  ShapeBenchmark PopResult();
  
  ShapeTester() {}

  ShapeTester(PhysicalVolume const *world_) {
    SetWorld(world_);
  }

  void SetWorld(PhysicalVolume const *world_) {
    world = world_;
  }

  void SetNumberOfPoints(const unsigned n_points_) {
    n_points = n_points_;
  }

  void SetRepetitions(const unsigned reps_) {
    reps = reps_;
  }

  void SetBias(const double bias_) {
    bias = bias_;
  }

  std::vector<ShapeBenchmark> Results() const {
    return results;
  }

private:
    
  void GenerateVolumePointers(PhysicalVolume const* /*vol*/);

};

#endif /* SHAPE_TESTER_H */