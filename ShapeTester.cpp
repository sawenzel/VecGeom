#include "mm_malloc.h"
#include "ShapeTester.h"
#include "GlobalDefs.h"
#include "Utils.h"

void ShapeTester::GenerateVolumePointers(PhysicalVolume const *vol) {

  VolumePointers pointers;
  pointers.fastgeom = vol;
  pointers.usolids = vol->GetAsUnplacedUSolid();
  pointers.root = vol->GetAsUnplacedROOTSolid();
  volumes.push_back(pointers);

  if (!vol->daughters || !vol->daughters->size()) return;
  for (auto d = vol->daughters->begin(); d != vol->daughters->end(); ++d) {
    GenerateVolumePointers(*d);
  }

}

void ShapeTester::Run() {

  ShapeBenchmark benchmark;

  // Allocate "particles" and output memory
  Vectors3DSOA points, dirs, interm_points, interm_dirs;
  points.alloc(n_points);
  dirs.alloc(n_points);
  // interm_points.alloc(n_points);
  // interm_dirs.alloc(n_points);
  double *steps = (double*) _mm_malloc(n_points*sizeof(double),
                                       ALIGNMENT_BOUNDARY);
  double *distances = (double*) _mm_malloc(n_points*sizeof(double),
                                           ALIGNMENT_BOUNDARY);
  for (int i = 0; i < n_points; ++i) steps[i] = Utils::kInfinity;

  // Extract shape representations and generate particles
  volumes.clear();
  GenerateVolumePointers(world);
  const int n_vols = volumes.size();
  world->fillWithRandomPoints(points, n_points);
  world->fillWithBiasedDirections(points, dirs, n_points, bias);

  // Run DistanceToIn benchmark
  StopWatch timer;
  timer.Start();
  for (int i = 0; i < reps; ++i) {
    for (int j = 0; j < n_vols; ++j) {
      volumes[j].fastgeom->DistanceToIn(points, dirs, steps, distances);
    }
  }
  timer.Stop();
  benchmark.fastgeom = timer.getDeltaSecs();

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
  ShapeBenchmark result = results.back();
  results.pop_back();
  return result;
}