#include "mm_malloc.h"
#include "ShapeTester.h"
#include "GlobalDefs.h"
#include "Utils.h"

void ShapeTester::GenerateVolumePointers(PhysicalVolume const *vol) {

  if (vol != world) {
    VolumePointers pointers;
    pointers.fastgeom = vol;
    pointers.usolids = vol->GetAsUnplacedUSolid();
    pointers.root = vol->GetAsUnplacedROOTSolid();
    volumes.push_back(pointers);
  }

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

  // Convert to USolids and ROOT representations
  std::vector<Vector3D> points_vec(n_points);
  std::vector<Vector3D> dirs_vec(n_points);
  points.toStructureOfVector3D(points_vec);
  dirs.toStructureOfVector3D(dirs_vec);

  if (verbose) std::cout << "Running benchmark for " << n_vols
                         << " volume(s) with " << reps << " repetitions... ";

  // Benchmark fastgeom
  StopWatch timer;
  timer.Start();
  for (int r = 0; r < reps; ++r) {
    for (int v = 0; v < n_vols; ++v) {
      volumes[v].fastgeom->DistanceToIn(points, dirs, steps, distances);
    }
  }
  timer.Stop();
  benchmark.fastgeom = timer.getDeltaSecs();

  // Benchmark USolids
  timer.Start();
  for (int r = 0; r < reps; ++r) {
    for (int v = 0; v < n_vols; ++v) {
      TransformationMatrix const *matrix = volumes[v].fastgeom->getMatrix();
      for (int p = 0; p < n_points; ++p) {
        Vector3D point_local, dir_local;
        matrix->MasterToLocal<1,-1>(points_vec[p], point_local);
        matrix->MasterToLocalVec<-1>(dirs_vec[p], dir_local);
        volumes[v].usolids->DistanceToIn(
            reinterpret_cast<UVector3 const&>(point_local),
            reinterpret_cast<UVector3 const&>(dir_local), steps[p]);
      }
    }
  }
  timer.Stop();
  benchmark.usolids = timer.getDeltaSecs();

  // Benchmark ROOT
  timer.Start();
  for (int r = 0; r < reps; ++r) {
    for (int v = 0; v < n_vols; ++v) {
      TransformationMatrix const *matrix = volumes[v].fastgeom->getMatrix();
      for (int p = 0; p < n_points; ++p) {
        Vector3D point_local, dir_local;
        matrix->MasterToLocal<1,-1>(points_vec[p], point_local);
        matrix->MasterToLocalVec<-1>(dirs_vec[p], dir_local);
        volumes[v].root->DistFromOutside(&point_local.x, &dir_local.x,
                                         3, steps[p], 0);
      }
    }
  }
  timer.Stop();
  benchmark.root = timer.getDeltaSecs();

  if (verbose) {
    std::cout << "Done." << std::endl;
    benchmark.Print();
  }

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