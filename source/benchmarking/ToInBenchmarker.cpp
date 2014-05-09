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

void ToInBenchmarker::BenchmarkAll() {

  printf("Running DistanceToIn and SafetyToIn benchmark for %i points for "
         "%i repetitions.\n", fPointCount, fRepetitions);

  PrepareBenchmark();

  std::stringstream outputLabels;
  outputLabels << "Specialized - Vectorized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized = AllocateDistance();
  Precision *const safetiesSpecialized = AllocateDistance();
  Precision *const distancesVectorized = AllocateDistance();
  Precision *const safetiesVectorized = AllocateDistance();
  Precision *const distancesUnspecialized = AllocateDistance();
  Precision *const safetiesUnspecialized = AllocateDistance();
#ifdef VECGEOM_USOLIDS
  Precision *const distancesUSolids = AllocateDistance();
  Precision *const safetiesUSolids = AllocateDistance();
  outputLabels << " - USolids";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateDistance();
  Precision *const safetiesRoot = AllocateDistance();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_CUDA
  Precision *const distancesCuda = AllocateDistance();
  Precision *const safetiesCuda = AllocateDistance();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
  RunSpecialized(distancesSpecialized, safetiesSpecialized);
  RunVectorized(distancesVectorized, safetiesVectorized);
  RunUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#ifdef VECGEOM_USOLIDS
  RunUSolids(distancesUSolids, safetiesUSolids);
#endif
#ifdef VECGEOM_ROOT
  RunRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_CUDA
  RunCuda(fPointPool->x(),     fPointPool->y(),     fPointPool->z(),
          fDirectionPool->x(), fDirectionPool->y(), fDirectionPool->z(),
          distancesCuda, safetiesCuda);
#endif

  if (fVerbose > 1) printf("Comparing DistanceToIn results:\n%s\n",
                           outputLabels.str().c_str());

  // Compare results
  int mismatches = 0;
  for (unsigned i = 0; i < fPointCount; ++i) {
    bool mismatch = false;
    std::stringstream mismatchOutput;
    if (fVerbose > 1) {
      mismatchOutput << distancesSpecialized[i] << " / "
                     << distancesVectorized[i] << " / "
                     << distancesUnspecialized[i];
    }
    if (abs(distancesSpecialized[i] - distancesVectorized[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesVectorized[i] == kInfinity)) {
      mismatch = true;
    }
    if (abs(distancesSpecialized[i] - distancesUnspecialized[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesUnspecialized[i] == kInfinity)) {
      mismatch = true;
    }
#ifdef VECGEOM_ROOT
    if (abs(distancesSpecialized[i] - distancesRoot[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesRoot[i] == 1e30)) {
      mismatch = true;
    }
    if (fVerbose > 1) mismatchOutput << " / " << distancesRoot[i];
#endif
#ifdef VECGEOM_USOLIDS
    if (abs(distancesSpecialized[i] - distancesUSolids[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesUSolids[i] == UUtils::kInfinity)) {
      mismatch = true;
    }
    if (fVerbose > 1) mismatchOutput << " / " << distancesUSolids[i];
#endif
#ifdef VECGEOM_CUDA
    if (abs(distancesSpecialized[i] - distancesCuda[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesCuda[i] == kInfinity)) {
      mismatch = true;
    }
    if (fVerbose > 1) mismatchOutput << " / " << distancesCuda[i];
#endif
    if (mismatch) {
      ++mismatches;
      if (fVerbose > 1) printf("%s\n", mismatchOutput.str().c_str());
    }
  }
  if (fVerbose > 0) {
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
  }

  // Clean up memory
  FreeDistance(distancesSpecialized);
  FreeDistance(distancesUnspecialized);
  FreeDistance(distancesVectorized);
#ifdef VECGEOM_USOLIDS
  FreeDistance(distancesUSolids);
#endif
#ifdef VECGEOM_ROOT
  FreeDistance(distancesRoot);
#endif
#ifdef VECGEOM_CUDA
  FreeDistance(distancesCuda);
#endif

  if (fVerbose > 1) printf("Comparing SafetyToIn results:\n%s\n",
                           outputLabels.str().c_str());

  // Compare results
  mismatches = 0;
  for (unsigned i = 0; i < fPointCount; ++i) {
    bool mismatch = false;
    std::stringstream mismatchOutput;
    if (fVerbose > 1) {
      mismatchOutput << safetiesSpecialized[i] << " / "
                     << safetiesVectorized[i] << " / "
                     << safetiesUnspecialized[i];
    }
    if (abs(safetiesSpecialized[i] - safetiesVectorized[i]) > kTolerance
        && !(safetiesSpecialized[i] == kInfinity &&
             safetiesVectorized[i] == kInfinity)) {
      mismatch = true;
    }
    if (abs(safetiesSpecialized[i] - safetiesUnspecialized[i]) > kTolerance
        && !(safetiesSpecialized[i] == kInfinity &&
             safetiesUnspecialized[i] == kInfinity)) {
      mismatch = true;
    }
#ifdef VECGEOM_ROOT
    if (abs(safetiesSpecialized[i] - safetiesRoot[i]) > kTolerance
        && !(safetiesSpecialized[i] == kInfinity &&
             safetiesRoot[i] == 1e30)) {
      mismatch = true;
    }
    if (fVerbose > 1) mismatchOutput << " / " << safetiesRoot[i];
#endif
#ifdef VECGEOM_USOLIDS
    if (abs(safetiesSpecialized[i] - safetiesUSolids[i]) > kTolerance
        && !(safetiesSpecialized[i] == kInfinity &&
             safetiesUSolids[i] == UUtils::kInfinity)) {
      mismatch = true;
    }
    if (fVerbose > 1) mismatchOutput << " / " << safetiesUSolids[i];
#endif
#ifdef VECGEOM_CUDA
    if (abs(safetiesSpecialized[i] - safetiesCuda[i]) > kTolerance
        && !(safetiesSpecialized[i] == kInfinity &&
             safetiesCuda[i] == kInfinity)) {
      mismatch = true;
    }
    if (fVerbose > 1) mismatchOutput << " / " << safetiesCuda[i];
#endif
    if (mismatch) {
      ++mismatches;
      if (fVerbose > 1) printf("%s\n", mismatchOutput.str().c_str());
    }
  }
  if (fVerbose) {
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
  }

  FreeDistance(safetiesSpecialized);
  FreeDistance(safetiesUnspecialized);
  FreeDistance(safetiesVectorized);
#ifdef VECGEOM_USOLIDS
  FreeDistance(safetiesUSolids);
#endif
#ifdef VECGEOM_ROOT
  FreeDistance(safetiesRoot);
#endif
#ifdef VECGEOM_CUDA
  FreeDistance(safetiesCuda);
#endif

}

void ToInBenchmarker::BenchmarkSpecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  Precision *const safeties = AllocateDistance();
  RunSpecialized(distances, safeties);
  FreeDistance(distances);
  FreeDistance(safeties);
}

void ToInBenchmarker::BenchmarkVectorized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  Precision *const safeties = AllocateDistance();
  RunVectorized(distances, safeties);
  FreeDistance(distances);
  FreeDistance(safeties);
}

void ToInBenchmarker::BenchmarkUnspecialized() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  Precision *const safeties = AllocateDistance();
  RunUnspecialized(distances, safeties);
  FreeDistance(distances);
  FreeDistance(safeties);
}

#ifdef VECGEOM_USOLIDS
void ToInBenchmarker::BenchmarkUSolids() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  Precision *const safeties = AllocateDistance();
  RunUSolids(distances, safeties);
  FreeDistance(distances);
  FreeDistance(safeties);
}
#endif

#ifdef VECGEOM_ROOT
void ToInBenchmarker::BenchmarkRoot() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  Precision *const safeties = AllocateDistance();
  RunRoot(distances, safeties);
  FreeDistance(distances);
  FreeDistance(safeties);
}
#endif

#ifdef VECGEOM_CUDA
void ToInBenchmarker::BenchmarkCuda() {
  PrepareBenchmark();
  Precision *const distances = AllocateDistance();
  Precision *const safeties = AllocateDistance();
  RunCuda(fPointPool->x(),     fPointPool->y(),     fPointPool->z(),
          fDirectionPool->x(), fDirectionPool->y(), fDirectionPool->z(),
          distances, safeties);
  FreeDistance(distances);
  FreeDistance(safeties);
}
#endif

void ToInBenchmarker::RunSpecialized(
    Precision *const distances, Precision *const safeties) const {
  printf("Running specialized benchmark...");
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
  const Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        const int p = index + i;
        safeties[i] = v->specialized()->SafetyToIn(
          (*fPointPool)[p]
        );
      }
    }
  }
  const Precision elapsedSafety = timer.Stop();
  printf(" Finished in %fs/%fs.\n", elapsedDistance, elapsedSafety);
}

void ToInBenchmarker::RunVectorized(
    Precision *const distances, Precision *const safeties) const {
  printf("Running specialized benchmark with vector interface...");
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
  const Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    SOA3D<Precision> points(&fPointPool->x(index), &fPointPool->y(index),
                            &fPointPool->z(index), fPointCount);
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      v->specialized()->SafetyToIn(points, safeties);
    }
  }
  const Precision elapsedSafety = timer.Stop();
  printf(" Finished in %f/%fs.\n", elapsedDistance, elapsedSafety);
}

void ToInBenchmarker::RunUnspecialized(
    Precision *const distances, Precision *const safeties) const {
  printf("Running unspecialized benchmark...");
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
  const Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    const int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(); v != fVolumes.end(); ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        const int p = index + i;
        safeties[i] = v->unspecialized()->SafetyToIn(
          (*fPointPool)[p]
        );
      }
    }
  }
  const Precision elapsedSafety = timer.Stop();
  printf(" Finished in %fs/%fs.\n", elapsedDistance, elapsedSafety);
}

#ifdef VECGEOM_USOLIDS
void ToInBenchmarker::RunUSolids(
    Precision *const distances, Precision *const safeties) const {
  printf("Running USolids benchmark...");
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
  const Precision elapsedDistance = timer.Stop();
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
        safeties[i] = v->usolids()->SafetyFromOutside(
          UVector3(point[0], point[1], point[2])
        );
      }
    }
  }
  const Precision elapsedSafety = timer.Stop();
  printf(" Finished in %fs/%fs.\n", elapsedDistance, elapsedSafety);
}
#endif

#ifdef VECGEOM_ROOT
void ToInBenchmarker::RunRoot(
    Precision *const distances, Precision *const safeties) const {
  printf("Running ROOT benchmark...");
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
  const Precision elapsedDistance = timer.Stop();
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
        safeties[i] = v->root()->Safety(&point[0], false);
      }
    }
  }
  const Precision elapsedSafety = timer.Stop();
  printf(" Finished in %fs/%fs.\n", elapsedDistance, elapsedSafety);
}
#endif

double* ToInBenchmarker::AllocateDistance() const {
  return (double*) _mm_malloc(fPointCount*sizeof(double), kAlignmentBoundary);
}

void ToInBenchmarker::FreeDistance(double *const distance) {
  if (distance) _mm_free(distance);
}

} // End namespace vecgeom