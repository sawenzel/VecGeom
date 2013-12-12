#ifndef SHAPE_TESTER_H
#define SHAPE_TESTER_H

#include <vector>
#include <iostream>
#include <iomanip>
#include "GeoManager.h"

typedef struct {
  double fast_geom;
  void Print() const {
    std::cout << "Benchmark:\n"
              << std::setw(2) << fast_geom
              << std::endl;
  }
} ShapeBenchmark;

class ShapeTester {

private:

  PhysicalVolume const *world;
  // std::vector<PhysicalVolume const*> volumes;
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

  ~ShapeTester() {
    // ClearVolumes();
  }

  // void ClearVolumes() {
  //   volumes.clear();
  // }

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

  // void AddShape(PhysicalVolume const *volume) {
  //   volumes.push_back(volume);
  // }

  std::vector<ShapeBenchmark> Results() const {
    return results;
  }

  // template <typename Shape, typename Parameters = ShapeParametersMap<Shape>>
  // void SetWorld(Parameters const *params, TransformationMatrix const *tm) {
  //   delete world;
  //   world = GeoManager::MakePlacedShape<Shape>(params, tm);
  // }

  // template <typename Shape, typename Parameters = ShapeParametersMap<Shape>>
  // void PlaceShape(Parameters const *params, TransformationMatrix const *tm) {
  //   PhysicalVolume *volume = GeoManager::MakePlacedShape<Shape>(params, tm);
  //   volumes.push_back(volume);
  //   world->AddDaughter(volume);
  // }

private:

  void DistanceToIn(PhysicalVolume const* /*vol*/,
                    Vectors3DSOA const& /*points*/,
                    Vectors3DSOA const& /*dirs*/,
                    double const* /*steps*/,
                    double* /*distances*/) const;

};

#endif /* SHAPE_TESTER_H */