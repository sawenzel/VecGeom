/// @file ToInBenchmarker.cpp
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "benchmarking/ToInBenchmarker.h"

#include "base/iterator.h"
#include "base/soa3d.h"
#include "base/stopwatch.h"
#include "base/transformation3d.h"
#include "volumes/logical_volume.h"
#include "volumes/placed_box.h"
#include "volumes/utilities/volume_utilities.h"

#ifdef VECGEOM_USOLIDS
#include "VUSolid.hh"
#include "UUtils.hh"
#include "UVector3.hh"
#endif

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

#include <random>
#include <sstream>

namespace vecgeom {

ToInBenchmarker::ToInBenchmarker(
    VPlacedVolume const *const world)
    : fWorld(world), fPointCount(1024), fPoolMultiplier(1), fRepetitions(1024),
      fBias(0.8), fPointPool(NULL), fDirectionPool(NULL), fStepMax(NULL) {}

ToInBenchmarker::~ToInBenchmarker() {
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  FreeDistance(fStepMax);
}

void ToInBenchmarker::SetPoolMultiplier(
    const unsigned poolMultiplier) {
  assert(poolMultiplier >= 1
         && "Pool multiplier for benchmarker must be >= 1.");
  fPoolMultiplier = poolMultiplier;
}

void ToInBenchmarker::GenerateVolumePointers(VPlacedVolume const *const vol) {

  fVolumes.emplace(fVolumes.end(), vol);

  for (auto i = vol->daughters().begin(), i_end = vol->daughters().end();
       i != i_end; ++i) {
    GenerateVolumePointers(*i);
  }

}

BenchmarkResult ToInBenchmarker::GenerateBenchmarkResult(
    const Precision elapsed, const BenchmarkType type) const {
  const BenchmarkResult benchmark = {
    .elapsed = elapsed,
    .type = type,
    .repetitions = fRepetitions,
    .volumes = static_cast<unsigned>(fVolumes.size()),
    .points = fPointCount,
    .bias = fBias
  };
  return benchmark;
}

void ToInBenchmarker::PrepareBenchmark() {

  // Allocate memory
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  if (fStepMax) delete fStepMax;
  fPointPool = new SOA3D<Precision>(fPointCount*fPoolMultiplier);
  fDirectionPool = new SOA3D<Precision>(fPointCount*fPoolMultiplier);
  fStepMax = AllocateDistance();
  for (unsigned i = 0; i < fPointCount; ++i) fStepMax[i] = kInfinity;

  // Generate pointers to representations in each geometry
  fVolumes.clear();
  GenerateVolumePointers(fWorld);

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  volumeutilities::FillUncontainedPoints(*fWorld, *fPointPool);
  volumeutilities::FillBiasedDirections(*fWorld, *fPointPool, fBias,
                                        *fDirectionPool);

  fPointPool->set_size(fPointCount*fPoolMultiplier);
  fDirectionPool->set_size(fPointCount*fPoolMultiplier);
}

std::list<BenchmarkResult> ToInBenchmarker::BenchmarkAll() {

  if (fVerbose > 0) std::cout << "Running DistanceToIn benchmark for "
                              << fPointCount << " points for " << fRepetitions
                              << " repetitions.\n";

  PrepareBenchmark();

  std::list<BenchmarkResult> results;
  std::stringstream outputLabels("Specialized / Vectorized / Unspecialized");

  // Allocate output memory
  Precision *const distancesSpecialized = AllocateDistance();
  Precision *const distancesVectorized = AllocateDistance();
  Precision *const distancesUnspecialized = AllocateDistance();
#ifdef VECGEOM_USOLIDS
  Precision *const distancesUSolids = AllocateDistance();
  outputLabels << " / USolids";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateDistance();
  outputLabels << " / ROOT";
#endif
#ifdef VECGEOM_CUDA
  Precision *const distancesCuda = AllocateDistance();
  outputLabels << " / CUDA";
#endif

  // Run all benchmarks
  results.push_back(RunSpecialized(distancesSpecialized));
  results.push_back(RunVectorized(distancesVectorized));
  results.push_back(RunUnspecialized(distancesUnspecialized));
#ifdef VECGEOM_USOLIDS
  results.push_back(RunUSolids(distancesUSolids));
#endif
#ifdef VECGEOM_ROOT
  results.push_back(RunRoot(distancesRoot));
#endif
#ifdef VECGEOM_CUDA
  results.push_back(
    GenerateBenchmarkResult(RunCuda(
      fPointPool->x(),     fPointPool->y(),     fPointPool->z(),
      fDirectionPool->x(), fDirectionPool->y(), fDirectionPool->z(),
      distancesCuda
    ), kCuda)
  );
#endif

  if (fVerbose > 1) printf("Printing mismatches if found.\n%s",
                           outputLabels.str().c_str());

  // Compare results
  int mismatches = 0;
  for (unsigned i = 0; i < fPointCount; ++i) {
    bool mismatch = false;
    std::stringstream mismatchOutput;
    if (fVerbose > 1) {
      mismatchOutput << distancesSpecialized[i];
    }
    if (abs(distancesSpecialized[i] - distancesVectorized[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesVectorized[i]  == kInfinity)) {
      mismatch = true;
      if (fVerbose > 1) mismatchOutput << " / " << distancesVectorized[i];
    }
#ifdef VECGEOM_ROOT
    if (abs(distancesSpecialized[i] - distancesRoot[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesRoot[i]  == 1e30)) {
      mismatch = true;
      if (fVerbose > 1) mismatchOutput << " / " << distancesRoot[i];
    }
#endif
#ifdef VECGEOM_USOLIDS
    if (abs(distancesSpecialized[i] - distancesUSolids[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesUSolids[i] == UUtils::kInfinity)) {
      mismatch = true;
      if (fVerbose > 1) mismatchOutput << " / " << distancesUSolids[i];
    }
#endif
#ifdef VECGEOM_CUDA
    if (abs(distancesSpecialized[i] - distancesCuda[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesCuda[i] == kInfinity)) {
      mismatch = true;
      if (fVerbose > 1) mismatchOutput << " / " << distancesCuda[i];
    }
#endif
    mismatches += mismatch;
    if (fVerbose > 1) printf("%s\n", mismatchOutput.str().c_str());
  }
  if (fVerbose) {
    std::cout << mismatches << " / " << fPointCount
              << " mismatches detected.\n";
  }

  // Clean up memory
  FreeDistance(distancesSpecialized);
  FreeDistance(distancesUnspecialized);
#ifdef VECGEOM_USOLIDS
  FreeDistance(distancesUSolids);
#endif
#ifdef VECGEOM_ROOT
  FreeDistance(distancesRoot);
#endif
#ifdef VECGEOM_CUDA
  FreeDistance(distancesCuda);
#endif

  return results;
}

BenchmarkResult ToInBenchmarker::BenchmarkSpecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  BenchmarkResult result = RunSpecialized(distances);
  FreeDistance(distances);
  return result;
}

BenchmarkResult ToInBenchmarker::BenchmarkVectorized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  BenchmarkResult result = RunVectorized(distances);
  FreeDistance(distances);
  return result;
}

BenchmarkResult ToInBenchmarker::BenchmarkUnspecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  BenchmarkResult result = RunUnspecialized(distances);
  FreeDistance(distances);
  return result;
}

#ifdef VECGEOM_USOLIDS
BenchmarkResult ToInBenchmarker::BenchmarkUSolids() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  BenchmarkResult result = RunUSolids(distances);
  FreeDistance(distances);
  return result;
}
#endif

#ifdef VECGEOM_ROOT
BenchmarkResult ToInBenchmarker::BenchmarkRoot() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  BenchmarkResult result = RunRoot(distances);
  FreeDistance(distances);
  return result;
}
#endif

#ifdef VECGEOM_CUDA
BenchmarkResult ToInBenchmarker::BenchmarkCuda() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  BenchmarkResult result = GenerateBenchmarkResult(RunCuda(
      fPointPool->x(),     fPointPool->y(),     fPointPool->z(),
      fDirectionPool->x(), fDirectionPool->y(), fDirectionPool->z(),
      distances), kCuda);
  FreeDistance(distances);
  return result;
}
#endif

BenchmarkResult ToInBenchmarker::RunSpecialized(
    Precision *const distances) const {
  if (fVerbose > 0) printf("Running specialized benchmark...");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        const int p = index + i;
        distances[i] = v->specialized()->DistanceToIn(
          (*fPointPool)[p], (*fDirectionPool)[p]
        );
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (fVerbose > 0) printf(" Finished in %fs.\n", elapsed);
  return GenerateBenchmarkResult(elapsed, kSpecialized);
}

BenchmarkResult ToInBenchmarker::RunVectorized(
    Precision *const distances) const {
  if (fVerbose > 0) {
    printf("Running specialized benchmark with vector interface...");
  }
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    SOA3D<Precision> points(&fPointPool->x(index), &fPointPool->y(index),
                            &fPointPool->z(index), fPointCount);
    SOA3D<Precision> directions(&fDirectionPool->x(index),
                                &fDirectionPool->y(index),
                                &fDirectionPool->z(index), fPointCount);
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      v->specialized()->DistanceToIn(points, directions, fStepMax, distances);
    }
  }
  const Precision elapsed = timer.Stop();
  if (fVerbose > 0) std::cout << " Finished in " << elapsed << "s.\n";
  return GenerateBenchmarkResult(elapsed, kVectorized);
}

BenchmarkResult ToInBenchmarker::RunUnspecialized(
    Precision *const distances) const {
  if (fVerbose > 0) printf("Running unspecialized benchmark...");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        const int p = index + i;
        distances[i] = v->unspecialized()->DistanceToIn(
          (*fPointPool)[p], (*fDirectionPool)[p]
        );
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (fVerbose > 0) printf(" Finished in %fs.\n", elapsed);
  return GenerateBenchmarkResult(elapsed, kUnspecialized);
}

#ifdef VECGEOM_USOLIDS
BenchmarkResult ToInBenchmarker::RunUSolids(
    Precision *const distances) const {
  if (fVerbose > 0) printf("Running USolids benchmark...");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      Transformation3D const *transformation =
          v->unspecialized()->transformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        const int p = index + i;
        const Vector3D<Precision> point =
            transformation->Transform((*fPointPool)[p]);
        const Vector3D<Precision> dir =
            transformation->TransformDirection((*fDirectionPool)[p]);
        distances[i] = v->usolids()->DistanceToIn(
          UVector3(point[0], point[1], point[2]),
          UVector3(dir[0], dir[1], dir[2])
        );
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (fVerbose > 0) printf(" Finished in %fs.\n", elapsed);
  return GenerateBenchmarkResult(elapsed, kUSolids);
}
#endif

#ifdef VECGEOM_ROOT
BenchmarkResult ToInBenchmarker::RunRoot(
    Precision *const distances) const {
  if (fVerbose > 0) printf("Running ROOT benchmark...");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      Transformation3D const *transformation =
          v->unspecialized()->transformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        const int p = index + i;
        Vector3D<Precision> point =
            transformation->Transform((*fPointPool)[p]);
        Vector3D<Precision> dir =
            transformation->TransformDirection((*fDirectionPool)[p]);
        distances[i] = v->root()->DistFromOutside(&point[0], &dir[0]);
      }
    }
  }
  const Precision elapsed = timer.Stop();
  if (fVerbose > 0) printf(" Finished in %fs.\n", elapsed);
  return GenerateBenchmarkResult(elapsed, kRoot);
}
#endif

double* ToInBenchmarker::AllocateDistance() const {
  return (double*) _mm_malloc(fPointCount*sizeof(double), kAlignmentBoundary);
}

void ToInBenchmarker::FreeDistance(double *const distance) {
  if (distance) _mm_free(distance);
}

} // End namespace vecgeom