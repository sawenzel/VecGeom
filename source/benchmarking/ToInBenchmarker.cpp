/// @file Benchmarker.cpp
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "benchmarking/Benchmarker.h"

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

Benchmarker::Benchmarker(
    VPlacedVolume const *const world)
    : fWorld(world), fPointCount(1024), fPoolMultiplier(1), fRepetitions(1024),
      fVerbosity(1), fToInBias(0.8), fInsideBias(0.5), fPointPool(NULL),
      fDirectionPool(NULL), fStepMax(NULL) {}

Benchmarker::~Benchmarker() {
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  FreeAligned(fStepMax);
}

void Benchmarker::SetPoolMultiplier(
    const unsigned poolMultiplier) {
  assert(poolMultiplier >= 1
         && "Pool multiplier for benchmarker must be >= 1.");
  fPoolMultiplier = poolMultiplier;
}

void Benchmarker::GenerateVolumePointers(VPlacedVolume const *const vol) {

  fVolumes.emplace(fVolumes.end(), vol);

  for (auto i = vol->daughters().begin(), i_end = vol->daughters().end();
       i != i_end; ++i) {
    GenerateVolumePointers(*i);
  }

}

BenchmarkResult Benchmarker::GenerateBenchmarkResult(
    const Precision elapsed, const EBenchmarkedMethod method,
    const EBenchmarkedLibrary library, const double bias) const {
  const BenchmarkResult benchmark = {
    .elapsed = elapsed,
    .method = method,
    .library = library,
    .repetitions = fRepetitions,
    .volumes = static_cast<unsigned>(fVolumes.size()),
    .points = fPointCount,
    .bias = bias
  };
  return benchmark;
}

void Benchmarker::RunBenchmark() {
  RunToInBenchmark();
}

void Benchmarker::RunToInBenchmark() {

  // Allocate memory
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  if (fStepMax) delete fStepMax;
  fPointPool = new SOA3D<Precision>(fPointCount*fPoolMultiplier);
  fDirectionPool = new SOA3D<Precision>(fPointCount*fPoolMultiplier);
  fStepMax = AllocateAligned();
  for (unsigned i = 0; i < fPointCount; ++i) fStepMax[i] = kInfinity;

  // Generate pointers to representations in each geometry
  fVolumes.clear();
  GenerateVolumePointers(fWorld);
  
  if (fVerbosity > 1) {
    printf("Found %lu volumes in world volume to be used for benchmarking.\n",
           fVolumes.size());
  }

  if (fVerbosity > 2) printf("Generating biased points...");

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  volumeUtilities::FillUncontainedPoints(*fWorld, *fPointPool);
  volumeUtilities::FillBiasedDirections(*fWorld, *fPointPool, fToInBias,
                                        *fDirectionPool);

  if (fVerbosity > 2) printf(" Done.\n");

  fPointPool->set_size(fPointCount*fPoolMultiplier);
  fDirectionPool->set_size(fPointCount*fPoolMultiplier);

  if (fVerbosity > 1) {
    printf("Running DistanceToIn and SafetyToIn benchmark for %i points for "
           "%i repetitions.\n", fPointCount, fRepetitions);
  }
  if (fVerbosity > 2) {
    printf("Vector instruction size is %i doubles.\n", kVectorSize);
    printf("Times are printed as DistanceToIn/Safety.\n");
  }

  std::stringstream outputLabels;
  outputLabels << "Specialized - Vectorized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized = AllocateAligned();
  Precision *const safetiesSpecialized = AllocateAligned();
  Precision *const distancesVectorized = AllocateAligned();
  Precision *const safetiesVectorized = AllocateAligned();
  Precision *const distancesUnspecialized = AllocateAligned();
  Precision *const safetiesUnspecialized = AllocateAligned();
#ifdef VECGEOM_USOLIDS
  Precision *const distancesUSolids = AllocateAligned();
  Precision *const safetiesUSolids = AllocateAligned();
  outputLabels << " - USolids";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateAligned();
  Precision *const safetiesRoot = AllocateAligned();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_CUDA
  Precision *const distancesCuda = AllocateAligned();
  Precision *const safetiesCuda = AllocateAligned();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
  RunToInSpecialized(distancesSpecialized, safetiesSpecialized);
  RunToInVectorized(distancesVectorized, safetiesVectorized);
  RunToInUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#ifdef VECGEOM_USOLIDS
  RunToInUSolids(distancesUSolids, safetiesUSolids);
#endif
#ifdef VECGEOM_ROOT
  RunToInRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_CUDA
  RunToInCuda(fPointPool->x(),     fPointPool->y(),     fPointPool->z(),
              fDirectionPool->x(), fDirectionPool->y(), fDirectionPool->z(),
              distancesCuda, safetiesCuda);
#endif

  if (fVerbosity > 1) printf("Comparing DistanceToIn results:\n");
  if (fVerbosity > 2) printf("%s\n", outputLabels.str().c_str());

  // Compare results
  int mismatches = 0;
  for (unsigned i = 0; i < fPointCount; ++i) {
    bool mismatch = false;
    std::stringstream mismatchOutput;
    if (fVerbosity > 2) {
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
    if (fVerbosity > 2) mismatchOutput << " / " << distancesRoot[i];
#endif
#ifdef VECGEOM_USOLIDS
    if (abs(distancesSpecialized[i] - distancesUSolids[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesUSolids[i] == UUtils::kInfinity)) {
      mismatch = true;
    }
    if (fVerbosity > 2) mismatchOutput << " / " << distancesUSolids[i];
#endif
#ifdef VECGEOM_CUDA
    if (abs(distancesSpecialized[i] - distancesCuda[i]) > kTolerance
        && !(distancesSpecialized[i] == kInfinity &&
             distancesCuda[i] == kInfinity)) {
      mismatch = true;
    }
    if (fVerbosity > 2) mismatchOutput << " / " << distancesCuda[i];
#endif
    if (mismatch) {
      ++mismatches;
      if (fVerbosity > 2) printf("%s\n", mismatchOutput.str().c_str());
    }
  }
  if (fVerbosity > 1) {
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
  }

  // Clean up memory
  FreeAligned(distancesSpecialized);
  FreeAligned(distancesUnspecialized);
  FreeAligned(distancesVectorized);
#ifdef VECGEOM_USOLIDS
  FreeAligned(distancesUSolids);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(distancesRoot);
#endif
#ifdef VECGEOM_CUDA
  FreeAligned(distancesCuda);
#endif

  if (fVerbosity > 1) printf("Comparing SafetyToIn results:\n");
  if (fVerbosity > 2) printf("%s\n", outputLabels.str().c_str());

  // Compare results
  mismatches = 0;
  for (unsigned i = 0; i < fPointCount; ++i) {
    bool mismatch = false;
    std::stringstream mismatchOutput;
    if (fVerbosity > 2) {
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
    if (fVerbosity > 2) mismatchOutput << " / " << safetiesRoot[i];
#endif
#ifdef VECGEOM_USOLIDS
    if (abs(safetiesSpecialized[i] - safetiesUSolids[i]) > kTolerance
        && !(safetiesSpecialized[i] == kInfinity &&
             safetiesUSolids[i] == UUtils::kInfinity)) {
      mismatch = true;
    }
    if (fVerbosity > 2) mismatchOutput << " / " << safetiesUSolids[i];
#endif
#ifdef VECGEOM_CUDA
    if (abs(safetiesSpecialized[i] - safetiesCuda[i]) > kTolerance
        && !(safetiesSpecialized[i] == kInfinity &&
             safetiesCuda[i] == kInfinity)) {
      mismatch = true;
    }
    if (fVerbosity > 2) mismatchOutput << " / " << safetiesCuda[i];
#endif
    if (mismatch) {
      ++mismatches;
      if (fVerbosity > 2) printf("%s\n", mismatchOutput.str().c_str());
    }
  }
  if (fVerbosity > 1) {
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
  }

  FreeAligned(safetiesSpecialized);
  FreeAligned(safetiesUnspecialized);
  FreeAligned(safetiesVectorized);
#ifdef VECGEOM_USOLIDS
  FreeAligned(safetiesUSolids);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(safetiesRoot);
#endif
#ifdef VECGEOM_CUDA
  FreeAligned(safetiesCuda);
#endif

}

void Benchmarker::RunToInSpecialized(
    Precision *const distances, Precision *const safeties) {
  if (fVerbosity > 1) printf("Running specialized benchmark...");
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
  if (fVerbosity > 1) {
    printf(" Finished in %fs/%fs (%fs/%fs per volume).\n",
           elapsedDistance, elapsedSafety,
           elapsedDistance/fVolumes.size(), elapsedSafety/fVolumes.size());
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkSpecialized, fToInBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkSpecialized, fToInBias
    )
  );
}

void Benchmarker::RunToInVectorized(
    Precision *const distances, Precision *const safeties) {
  if (fVerbosity > 1) {
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
  if (fVerbosity > 1) {
    printf(" Finished in %fs/%fs (%fs/%fs per volume).\n", elapsedDistance,
           elapsedSafety, elapsedDistance/fVolumes.size(),
           elapsedSafety/fVolumes.size());
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkVectorized, fToInBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkVectorized, fToInBias
    )
  );
}

void Benchmarker::RunToInUnspecialized(
    Precision *const distances, Precision *const safeties) {
  if (fVerbosity > 1) printf("Running unspecialized benchmark...");
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
  if (fVerbosity > 1) {
    printf(" Finished in %fs/%fs (%fs/%fs per volume).\n", elapsedDistance,
           elapsedSafety, elapsedDistance/fVolumes.size(),
           elapsedSafety/fVolumes.size());
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkUnspecialized,
      fToInBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkUnspecialized, fToInBias
    )
  );
}

#ifdef VECGEOM_USOLIDS
void Benchmarker::RunToInUSolids(
    Precision *const distances, Precision *const safeties) {
  if (fVerbosity > 1) printf("Running USolids benchmark...");
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
  if (fVerbosity > 1) {
    printf(" Finished in %fs/%fs (%fs/%fs per volume).\n", elapsedDistance,
           elapsedSafety, elapsedDistance/fVolumes.size(),
           elapsedSafety/fVolumes.size());
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkUSolids, fToInBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkUSolids, fToInBias
    )
  );
}
#endif

#ifdef VECGEOM_ROOT
void Benchmarker::RunToInRoot(
    Precision *const distances, Precision *const safeties) {
  if (fVerbosity > 1) printf("Running ROOT benchmark...");
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
  if (fVerbosity > 1) {
    printf(" Finished in %fs/%fs (%fs/%fs per volume).\n", elapsedDistance,
           elapsedSafety, elapsedDistance/fVolumes.size(),
           elapsedSafety/fVolumes.size());
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkRoot, fToInBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkRoot, fToInBias
    )
  );
}
#endif

double* Benchmarker::AllocateAligned() const {
  return (double*) _mm_malloc(fPointCount*sizeof(double), kAlignmentBoundary);
}

void Benchmarker::FreeAligned(double *const distance) {
  if (distance) _mm_free(distance);
}

} // End namespace vecgeom