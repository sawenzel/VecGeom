#include "mm_malloc.h"
#include "ShapeTester.h"
#include "GlobalDefs.h"
#include "Utils.h"

void ShapeTester::DistanceToIn(PhysicalVolume const *vol,
                               Vectors3DSOA const &points,
                               Vectors3DSOA const &dirs, double const *steps,
                               double *distances) const {
  vol->DistanceToIn(points, dirs, steps, distances);
  for (auto d = vol->daughters->begin(); d != vol->daughters->end(); ++d) {
    DistanceToIn(*d, points, dirs, steps, distances);
  }
}

void ShapeTester::Run() {

  ShapeBenchmark benchmark;

  // Allocate "particles" with coordinates and directions
  Vectors3DSOA points, dirs, interm_points, interm_dirs;
  points.alloc(n_points);
  dirs.alloc(n_points);
  // interm_points.alloc(n_points);
  // interm_dirs.alloc(n_points);

  // Allocate output memory
  double *steps = (double*) _mm_malloc(n_points*sizeof(double),
                                       ALIGNMENT_BOUNDARY);
  double *distances = (double*) _mm_malloc(n_points*sizeof(double),
                                           ALIGNMENT_BOUNDARY);
  for (int i = 0; i < n_points; ++i) steps[i] = Utils::kInfinity;

  // Generate points and directions
  world->fillWithRandomPoints(points, n_points);
  world->fillWithBiasedDirections(points, dirs, n_points, bias);

  // Run DistanceToIn benchmark
  StopWatch timer;
  timer.Start();
  for (int i = 0; i < reps; ++i) {
    world->DistanceToIn(points, dirs, steps, distances);
  }
  timer.Stop();
  benchmark.fast_geom = timer.getDeltaSecs();

  // Clean up memory
  points.dealloc();
  dirs.dealloc();
  // interm_points.dealloc();
  // interm_dirs.dealloc();
  _mm_free(distances);
  _mm_free(steps);

  results.push_back(benchmark);
}

ShapeBenchmark ShapeTester::PopResult() {
  const ShapeBenchmark result = results.back();
  results.pop_back();
  return result;
}