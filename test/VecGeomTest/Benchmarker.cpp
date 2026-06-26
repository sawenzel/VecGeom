/// \file Benchmarker.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "Benchmarker.h"

#include "VecGeom/base/Assert.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Stopwatch.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/PlacedBox.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/volumes/kernel/TessellatedImplementation.h"
#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Backend.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/management/CudaManager.h"
#endif

#ifdef VECGEOM_GEANT4
#endif

#include <random>
#include <sstream>
#include <utility>
#include <atomic>

namespace vecgeom {

Benchmarker::Benchmarker() : Benchmarker(NULL) {}

Benchmarker::Benchmarker(VPlacedVolume const *const world, bool benchmarkTop)
    : fPointCount(1024), fPoolMultiplier(1), fRepetitions(1024), fMeasurementCount(1), fVerbosity(1), fToInBias(0.8),
      fInsideBias(0.5), fPointPool(NULL), fDirectionPool(NULL), fStepMax(NULL), fTolerance(kTolerance),
      fBenchmarkTop(benchmarkTop)
#ifdef VECGEOM_ROOT
      ,
      fOkToRunROOT(true)
#endif
#ifdef VECGEOM_GEANT4
      ,
      fOkToRunG4(true)
#endif
{
  SetWorld(world);
}

Benchmarker::~Benchmarker()
{
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  if (fStepMax) FreeAligned(fStepMax);
}

void Benchmarker::SetWorld(VPlacedVolume const *const world)
{
  fVolumes.clear();
  fWorld = world;
  if (!world) return;
  GenerateVolumePointers(fWorld);
  if (fVerbosity > 2) {
    printf("Found %zu volumes in world volume to be used for benchmarking.\n", fVolumes.size());
  }
}

void Benchmarker::SetPoolMultiplier(const unsigned poolMultiplier)
{
  VECGEOM_VALIDATE(poolMultiplier >= 1, << "Pool multiplier for benchmarker must be >= 1.");
  fPoolMultiplier = poolMultiplier;
}

std::list<BenchmarkResult> Benchmarker::PopResults()
{
  std::list<BenchmarkResult> results = fResults;
  fResults.clear();
  return results;
}

void Benchmarker::GenerateVolumePointers(VPlacedVolume const *const vol)
{
  if (fBenchmarkTop) {
    fVolumes.emplace_back(vol);
    VECGEOM_ASSERT(fVolumes.size() == 1);
    return;
  }
  for (auto i = vol->GetDaughters().begin(), iEnd = vol->GetDaughters().end(); i != iEnd; ++i) {

    // this is pretty tricky
    // this line causes the implicit conversion of a vecgeom shape
    // to the corresponding ROOT/G4 shapes via the VolumePointers
    // constructor
    fVolumes.push_back(*i);
// can check now the property of the conversions of *i
#ifdef VECGEOM_ROOT
    if (fVolumes.back().ROOT() == NULL) {
      fOkToRunROOT = false;
      std::cerr << "disabling ROOT\n";
    }
#endif
#ifdef VECGEOM_GEANT4
    if (fVolumes.back().Geant4() == NULL) {
      fOkToRunG4 = false;
      std::cerr << "disabling G4\n";
    }
#endif

    GenerateVolumePointers(*i);
  }
}

BenchmarkResult Benchmarker::GenerateBenchmarkResult(Precision elapsed, const EBenchmarkedMethod method,
                                                     const EBenchmarkedLibrary library, const Precision bias) const
{
  const BenchmarkResult benchmark = {.elapsed     = elapsed,
                                     .method      = method,
                                     .library     = library,
                                     .repetitions = fRepetitions,
                                     .volumes     = static_cast<unsigned>(fVolumes.size()),
                                     .points      = fPointCount,
                                     .bias        = bias};
  return benchmark;
}

int Benchmarker::CompareDistances(SOA3D<Precision> *points, SOA3D<Precision> *directions,
                                  Precision const *const specialized, Precision const *const unspecialized,
#ifdef VECGEOM_ROOT
                                  Precision const *const root,
#endif
#ifdef VECGEOM_GEANT4
                                  Precision const *const geant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                  Precision const *const cuda,
#endif
                                  char const *const method)
{

  fProblematicRays.clear();

  int mismatches = 0;
  std::stringstream outputLabelsStream;
  outputLabelsStream << "Specialized / Unspecialized";
#ifdef VECGEOM_ROOT
  if (fOkToRunROOT) outputLabelsStream << " / ROOT";
#endif
#ifdef VECGEOM_GEANT4
  if (fOkToRunG4) outputLabelsStream << " / Geant4";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  outputLabelsStream << " / CUDA";
#endif

  if (fPoolMultiplier == 1 && fVerbosity > 0) {

    printf("Comparing %s results...\n", method);
    if (fVerbosity > 2) printf("%s\n", outputLabelsStream.str().c_str());

    // Compare results
    for (unsigned i = 0; i < fPointCount; ++i) {
      bool mismatch = false;
      std::stringstream mismatchOutput;
      if (fVerbosity > 2) {
        mismatchOutput << specialized[i] << " / " << unspecialized[i];
      }
      if (!(specialized[i] == kInfLength && unspecialized[i] == kInfLength) &&
          std::fabs(specialized[i] - unspecialized[i]) > fTolerance) {
        mismatch = true;
        if (fVerbosity > 2) mismatchOutput << " (" << unspecialized[i] - specialized[i] << ") ";
      }
#ifdef VECGEOM_ROOT
      if (fOkToRunROOT) {
        // The miss condition 'root[i]==1e30' does not hold for scaled shape,
        // where
        // the returned distance is scaled with respect to the unscaled value
        if (fVerbosity > 2) mismatchOutput << " / " << root[i];
        if (std::fabs(specialized[i] - root[i]) > fTolerance && !(specialized[i] == kInfLength && root[i] > 1e20)) {
          mismatch = true;
          if (fVerbosity > 2) mismatchOutput << " (" << root[i] - specialized[i] << ") ";
        }
      }
#endif
#ifdef VECGEOM_GEANT4
      if (fOkToRunG4) {
        if (geant4) {
          if (fVerbosity > 2) mismatchOutput << " / " << geant4[i];
          if (!(specialized[i] == kInfLength && geant4[i] == ::kInfinity) &&
              std::fabs(specialized[i] - geant4[i]) > fTolerance) {
            mismatch = true;
            if (fVerbosity > 2) mismatchOutput << " (" << geant4[i] - specialized[i] << ") ";
          }
        }
      }
#endif
#ifdef VECGEOM_ENABLE_CUDA
      if (fVerbosity > 2) mismatchOutput << " / " << cuda[i];
      if (!(specialized[i] == kInfLength && cuda[i] == kInfLength) &&
          std::fabs(specialized[i] - cuda[i]) > fTolerance) {
        mismatch = true;
        if (fVerbosity > 2) mismatchOutput << " (" << cuda[i] - specialized[i] << ") ";
      }
#endif
      mismatches += mismatch;

      if (mismatch) {
        fProblematicRays.push_back(std::pair<Vector3D<Precision>, Vector3D<Precision>>(
            Vector3D<Precision>(points->x(i), points->y(i), points->z(i)),
            Vector3D<Precision>(directions->x(i), directions->y(i), directions->z(i))));
      }

      if ((mismatch && fVerbosity > 2) || fVerbosity > 4) {
        printf("Point (%.30f, %.30f, %.30f)", points->x(i), points->y(i), points->z(i));
        if (directions != NULL) {
          printf(", Direction (%.30f, %.30f, %.30f)", directions->x(i), directions->y(i), directions->z(i));
        }
        printf(": ");
      }

      if ((mismatch && fVerbosity > 2) || fVerbosity > 3) {
        printf("%s\n", mismatchOutput.str().c_str());
      }
    }
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
  }
  return mismatches;
}

int Benchmarker::CheckDistancesFromBoundary(Precision expected, SOA3D<Precision> *points, SOA3D<Precision> *directions,
                                            Precision const *const specialized, Precision const *const unspecialized,
#ifdef VECGEOM_ROOT
                                            Precision const *const root,
#endif
#ifdef VECGEOM_GEANT4
                                            Precision const *const geant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                            Precision const *const cuda,
#endif
                                            char const *const method)
{

  fProblematicRays.clear();

  int mismatches = 0;
  std::stringstream outputLabelsStream;
  outputLabelsStream << "Specialized / Unspecialized";
#ifdef VECGEOM_ROOT
  if (fOkToRunROOT) outputLabelsStream << " / ROOT";
#endif
#ifdef VECGEOM_GEANT4
  if (fOkToRunG4) outputLabelsStream << " / Geant4";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  outputLabelsStream << " / CUDA";
#endif

  if (fPoolMultiplier == 1 && fVerbosity > 0) {

    printf("Comparing %s results...\n", method);
    if (fVerbosity > 2) printf("%s\n", outputLabelsStream.str().c_str());

    // Compare results
    for (unsigned i = 0; i < fPointCount; ++i) {
      bool mismatch = false;
      std::stringstream mismatchOutput;
      if (fVerbosity > 2) {
        mismatchOutput << specialized[i] << " / " << unspecialized[i];
      }
      if (std::fabs(specialized[i] - unspecialized[i]) > fTolerance &&
          !(specialized[i] == kInfLength && unspecialized[i] == kInfLength)) {
        mismatch = true;
      }
#ifdef VECGEOM_ROOT
      if (fOkToRunROOT) {
        // The miss condition 'root[i]==1e30' does not hold for scaled shape,
        // where
        // the returned distance is scaled with respact to the unscaled value
        if (std::fabs(specialized[i] - root[i]) > fTolerance && !(specialized[i] == kInfLength && root[i] > 1e20)) {
          // NOT ANALYSING DIFFERENCE TO ROOT HERE
          // SINCE HARD CHECK IS GIVEN BY "expected" value
          // mismatch = true;
        }
        if (fVerbosity > 2) mismatchOutput << " / " << root[i];
      }
#endif
#ifdef VECGEOM_GEANT4
      if (fOkToRunG4) {
        if (geant4) {
          if (std::fabs(specialized[i] - geant4[i]) > fTolerance &&
              !(specialized[i] == kInfLength && geant4[i] == ::kInfinity)) {
            // NOT ANALYSING DIFFERENCE TO G4 HERE FOR MOMENT
            // SINCE HARD CHECK IS GIVEN BY "expected" value
            // mismatch = true;
          }
          if (fVerbosity > 2) mismatchOutput << " / " << geant4[i];
        }
      }
#endif
#ifdef VECGEOM_ENABLE_CUDA
      if (std::fabs(specialized[i] - cuda[i]) > fTolerance &&
          !(specialized[i] == kInfLength && cuda[i] == kInfLength)) {
        mismatch = true;
      }
      if (fVerbosity > 2) mismatchOutput << " / " << cuda[i];
#endif
      mismatches += mismatch;

      if (mismatch) {
        fProblematicRays.push_back(std::pair<Vector3D<Precision>, Vector3D<Precision>>(
            Vector3D<Precision>(points->x(i), points->y(i), points->z(i)),
            Vector3D<Precision>(directions->x(i), directions->y(i), directions->z(i))));
      }

      if ((mismatch && fVerbosity > 2) || fVerbosity > 4) {
        printf("Point (%.30f, %.30f, %.30f)", points->x(i), points->y(i), points->z(i));
        if (directions != NULL) {
          printf(", Direction (%.30f, %.30f, %.30f)", directions->x(i), directions->y(i), directions->z(i));
        }
        printf(": ");
      }

      if ((mismatch && fVerbosity > 2) || fVerbosity > 3) {
        printf("%s\n", mismatchOutput.str().c_str());
      }
    }
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
  }
  return mismatches;
}

int Benchmarker::CompareSafeties(SOA3D<Precision> *points, SOA3D<Precision> *directions,
                                 Precision const *const specialized, Precision const *const unspecialized,
#ifdef VECGEOM_ROOT
                                 Precision const *const root,
#endif
#ifdef VECGEOM_GEANT4
                                 Precision const *const geant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                 Precision const *const cuda,
#endif
                                 char const *const method) const
{

  int mismatches = 0;

  std::stringstream outputLabelsStream;

  outputLabelsStream << "Specialized / Unspecialized";
#ifdef VECGEOM_ROOT
  if (fOkToRunROOT) outputLabelsStream << " / ROOT";
#endif
#ifdef VECGEOM_GEANT4
  if (fOkToRunG4) outputLabelsStream << " / Geant4";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  outputLabelsStream << " / CUDA";
#endif
  if (fPoolMultiplier == 1 && fVerbosity > 0) {

    printf("Comparing %s results...\n", method);
    if (fVerbosity > 2) printf("%s\n", outputLabelsStream.str().c_str());

    // Compare results
    int worse = 0;
    for (unsigned i = 0; i < fPointCount; ++i) {
      bool mismatch = false;
      bool better   = true;
      std::stringstream mismatchOutput;
      if (fVerbosity > 2) {
        mismatchOutput << specialized[i] << " / " << unspecialized[i];
      }
      if (!(specialized[i] == kInfLength && unspecialized[i] == kInfLength) &&
          std::fabs(specialized[i] - unspecialized[i]) > kTolerance) {
        mismatch = true;
        better &= specialized[i] >= unspecialized[i] - kTolerance;
      }
#ifdef VECGEOM_ROOT
      if (fOkToRunROOT) {
        if (std::fabs(specialized[i] - root[i]) > kTolerance && !(specialized[i] == kInfLength && root[i] == 1e30)) {
          // mismatch = true;
          better &= specialized[i] >= root[i] - kTolerance;
        }
        if (fVerbosity > 2) mismatchOutput << " / " << root[i];
      }
#endif
#ifdef VECGEOM_GEANT4
      if (fOkToRunG4) {
        if (geant4) {
          if (std::fabs(specialized[i] - geant4[i]) > kTolerance &&
              !(specialized[i] == kInfLength && geant4[i] == ::kInfinity)) {
            //  mismatch = true;
            better &= specialized[i] >= geant4[i] - kTolerance;
          }
          if (fVerbosity > 2) mismatchOutput << " / " << geant4[i];
        }
      }
#endif
#ifdef VECGEOM_ENABLE_CUDA
      if (std::fabs(specialized[i] - cuda[i]) > kTolerance &&
          !(specialized[i] == kInfLength && cuda[i] == kInfLength)) {
        mismatch = true;
        better &= specialized[i] >= cuda[i] - kTolerance;
      }
      if (fVerbosity > 2) mismatchOutput << " / " << cuda[i];
#endif
      mismatches += mismatch;
      worse += !better;

      if ((!better && fVerbosity > 2) || fVerbosity > 4) {
        printf("Point (%f, %f, %f)", points->x(i), points->y(i), points->z(i));
        if (directions != NULL) {
          printf(", Direction (%f, %f, %f)", directions->x(i), directions->y(i), directions->z(i));
        }
        printf(": ");
      }

      if ((!better && fVerbosity > 2) || fVerbosity > 3) {
        printf("%s\n", mismatchOutput.str().c_str());
      }
    }
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
    printf("%i / %i worse safeties detected.\n", worse, fPointCount);
  }

  return mismatches;
}

int Benchmarker::CheckSafetiesOnBoundary(SOA3D<Precision> *points, SOA3D<Precision> *directions,
                                         Precision const *const specialized, Precision const *const unspecialized,
#ifdef VECGEOM_ROOT
                                         Precision const *const root,
#endif
#ifdef VECGEOM_GEANT4
                                         Precision const *const geant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                         Precision const *const cuda,
#endif
                                         char const *const method) const
{

  int mismatches = 0;

  std::stringstream outputLabelsStream;

  outputLabelsStream << "Specialized / Unspecialized";
#ifdef VECGEOM_ROOT
  if (fOkToRunROOT) outputLabelsStream << " / ROOT";
#endif
#ifdef VECGEOM_GEANT4
  if (fOkToRunG4) outputLabelsStream << " / Geant4";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  outputLabelsStream << " / CUDA";
#endif
  if (fPoolMultiplier == 1 && fVerbosity > 0) {

    printf("Comparing %s results...\n", method);
    if (fVerbosity > 2) printf("%s\n", outputLabelsStream.str().c_str());

    // Compare results
    int worse = 0;
    for (unsigned i = 0; i < fPointCount; ++i) {
      bool mismatch = false;
      bool better   = true;
      std::stringstream mismatchOutput;
      if (fVerbosity > 2) {
        mismatchOutput << specialized[i] << " / " << unspecialized[i];
      }
      if (!(specialized[i] == kInfLength && unspecialized[i] == kInfLength) &&
          std::fabs(specialized[i] - unspecialized[i]) > kTolerance) {
        mismatch = true;
      }
#ifdef VECGEOM_ROOT
      if (fOkToRunROOT) {
        if (std::fabs(specialized[i] - root[i]) > kTolerance && !(specialized[i] == kInfLength && root[i] == 1e30)) {
          // mismatch = true;
          // better &= specialized[i] >= root[i];
        }
        if (fVerbosity > 2) mismatchOutput << " / " << root[i];
      }
#endif
#ifdef VECGEOM_GEANT4
      if (fOkToRunG4) {
        if (geant4) {
          if (std::fabs(specialized[i] - geant4[i]) > kTolerance &&
              !(specialized[i] == kInfLength && geant4[i] == ::kInfinity)) {
            // mismatch = true;
            // better &= specialized[i] >= geant4[i];
          }
          if (fVerbosity > 2) mismatchOutput << " / " << geant4[i];
        }
      }
#endif
#ifdef VECGEOM_ENABLE_CUDA
      if (std::fabs(specialized[i] - cuda[i]) > kTolerance &&
          !(specialized[i] == kInfLength && cuda[i] == kInfLength)) {
        mismatch = true;
        // better &= specialized[i] >= cuda[i];
      }
      if (fVerbosity > 2) mismatchOutput << " / " << cuda[i];
#endif
      mismatches += mismatch;
      worse += !better;

      if ((!better && fVerbosity > 2) || fVerbosity > 4) {
        printf("Point (%f, %f, %f)", points->x(i), points->y(i), points->z(i));
        if (directions != NULL) {
          printf(", Direction (%f, %f, %f)", directions->x(i), directions->y(i), directions->z(i));
        }
        printf(": ");
      }

      if ((!better && fVerbosity > 2) || fVerbosity > 3) {
        printf("%s\n", mismatchOutput.str().c_str());
      }
    }
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);
    printf("%i / %i worse safeties detected.\n", worse, fPointCount);
  }

  return mismatches;
}

int Benchmarker::RunBenchmark()
{
  VECGEOM_ASSERT(fWorld != nullptr);
  int errorcode = 0;
// erase possible overhead counter accumulations of VeGeomTesselated implemenntation
#ifndef VECCORE_CUDA
  reset_counters_ARGH();
#endif // VECORE_CUDA
  errorcode += RunInsideBenchmark();
  errorcode += RunToInBenchmark();
  errorcode += RunToOutBenchmark();
  if (fMeasurementCount == 1) errorcode += CompareMetaInformation();
  return (errorcode) ? 1 : 0;
}

int Benchmarker::RunInsideBenchmark()
{
  int mismatches       = 0;
  int insidemismatches = 0;

  VECGEOM_ASSERT(fWorld);

  if (fVerbosity > 0) {
    printf("Running Contains and Inside benchmark for %i points for "
           "%i repetitions.\n",
           fPointCount, fRepetitions);
  }

  if (fPointPool) delete fPointPool;
  fPointPool = new SOA3D<Precision>(fPointCount * fPoolMultiplier);

  if (fVerbosity > 1) printf("Generating points with bias %f... ", fInsideBias);
  Stopwatch timer;
  timer.Start();
  volumeUtilities::FillContainedPoints(*fWorld, fInsideBias, *fPointPool, true, fBenchmarkTop);
  if (fVerbosity > 1) printf("Done in %f s.\n", timer.Stop());

  std::stringstream outputLabelsContains, outputLabelsInside;
  outputLabelsContains << "Specialized - Unspecialized";
  outputLabelsInside << "Specialized - Unspecialized";

  // Allocate memory
  bool *const containsSpecialized     = AllocateAligned<bool>();
  bool *const containsUnspecialized   = AllocateAligned<bool>();
  Inside_t *const insideSpecialized   = AllocateAligned<Inside_t>();
  Inside_t *const insideUnspecialized = AllocateAligned<Inside_t>();
#ifdef VECGEOM_ROOT
  bool *const containsRoot = AllocateAligned<bool>();
  if (fOkToRunROOT) outputLabelsContains << " - ROOT";
#endif
#ifdef VECGEOM_GEANT4
  ::EInside *const insideGeant4 = AllocateAligned<::EInside>();
  if (fOkToRunG4) outputLabelsInside << " - Geant4";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  bool *const containsCuda   = AllocateAligned<bool>();
  Inside_t *const insideCuda = AllocateAligned<Inside_t>();
  outputLabelsContains << " - CUDA";
  outputLabelsInside << " - CUDA";
#endif

  // Run all benchmarks
#ifndef VECCORE_CUDA
  enable_counters_ARGH();
#endif
  for (unsigned int i = 0; i < fMeasurementCount; ++i) {
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunInsideBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunInsideBenchmark, __itt_null, __itt_null, __itt_RunInsideSpecialized);
#endif
    RunInsideSpecialized(containsSpecialized, insideSpecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunInsideBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunInsideBenchmark, __itt_null, __itt_null, __itt_RunInsideUnspecialized);
#endif
    RunInsideUnspecialized(containsUnspecialized, insideUnspecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunInsideBenchmark);
#endif
#ifdef VECGEOM_GEANT4
    if (fOkToRunG4) RunInsideGeant4(insideGeant4);
#endif
#ifdef VECGEOM_ROOT
    if (fOkToRunROOT) RunInsideRoot(containsRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
    RunInsideCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), containsCuda, insideCuda);
#endif
  }
#ifndef VECCORE_CUDA
  disable_counters_ARGH();
#endif

  if (fPoolMultiplier == 1 && fVerbosity > 0) {

    // Compare Contains results
    printf("Comparing Contains results:\n");
    if (fVerbosity > 2) printf("%s\n", outputLabelsContains.str().c_str());

    // Compare results
    for (unsigned i = 0; i < fPointCount; ++i) {
      bool mismatch = false;

      // fProblematicContainPoints.push_back( fPointPool->operator[](i) );

      std::stringstream mismatchOutput;
      if (fVerbosity > 2) {
        mismatchOutput << containsSpecialized[i] << " / " << containsUnspecialized[i];
      }
      if (containsSpecialized[i] != containsUnspecialized[i]) mismatch = true;
#ifdef VECGEOM_ROOT
      if (fOkToRunROOT) {
        if (containsSpecialized[i] != containsRoot[i]) mismatch = true;
        if (fVerbosity > 2) mismatchOutput << " / " << containsRoot[i];
      }
#endif
#ifdef VECGEOM_GEANT4
      if (fOkToRunG4 && fVerbosity > 2)
        mismatchOutput << " / " << (insideGeant4[i] == EInside::kInside || insideGeant4[i] == EInside::kSurface);
#endif
#ifdef VECGEOM_ENABLE_CUDA
      if (containsSpecialized[i] != containsCuda[i]) mismatch = true;
      if (fVerbosity > 2) mismatchOutput << " / " << containsCuda[i];
#endif
      mismatches += mismatch;
      if ((mismatch && fVerbosity > 2) || fVerbosity > 4) {
        printf("Point (%f, %f, %f): ", fPointPool->x(i), fPointPool->y(i), fPointPool->z(i));

        // store point for later inspection
        fProblematicContainPoints.push_back(fPointPool->operator[](i));
      }
      if ((mismatch && fVerbosity > 2) || fVerbosity > 3) {
        printf("%s\n", mismatchOutput.str().c_str());
      }
    }
    if (fVerbosity > 2 && mismatches > 100) {
      printf("%s\n", outputLabelsContains.str().c_str());
    }
    printf("%i / %i mismatches detected.\n", mismatches, fPointCount);

    // Compare Inside results

    printf("Comparing Inside results:\n");
    if (fVerbosity > 2) printf("%s\n", outputLabelsInside.str().c_str());

    for (unsigned i = 0; i < fPointCount; ++i) {
      bool mismatch = false;
      std::stringstream mismatchOutput;
      if (fVerbosity > 2) {
        mismatchOutput << insideSpecialized[i] << " / " << insideUnspecialized[i];
      }
      if (insideSpecialized[i] != insideUnspecialized[i]) mismatch = true;
#ifdef VECGEOM_GEANT4
      if (fOkToRunG4) {
        if (!((insideSpecialized[i] == EInside::kInside && insideGeant4[i] == ::kInside) ||
              (insideSpecialized[i] == EInside::kOutside && insideGeant4[i] == ::kOutside) ||
              (insideSpecialized[i] == EInside::kSurface && insideGeant4[i] == ::kSurface))) {
          mismatch = true;
        }
        if (fVerbosity > 2) mismatchOutput << " / " << insideGeant4[i];
      }
#endif
#ifdef VECGEOM_ENABLE_CUDA
      if (insideSpecialized[i] != insideCuda[i]) mismatch = true;
      if (fVerbosity > 2) mismatchOutput << " / " << insideCuda[i];
#endif
      insidemismatches += mismatch;
      if ((mismatch && fVerbosity > 2) || fVerbosity > 4) {
        printf("Point (%f, %f, %f): ", *(fPointPool->x() + i), *(fPointPool->y() + i), fPointPool->z(i));
      }
      if ((mismatch && fVerbosity > 2) || fVerbosity > 3) {

        fProblematicContainPoints.push_back(fPointPool->operator[](i));
        printf("%s\n", mismatchOutput.str().c_str());
      }
    }
    if (fVerbosity > 2 && mismatches > 100) {
      printf("%s\n", outputLabelsInside.str().c_str());
    }
    printf("%i / %i mismatches detected.\n", insidemismatches, fPointCount);
  }

  // Clean up memory
  FreeAligned(containsSpecialized);
  FreeAligned(containsUnspecialized);
  FreeAligned(insideSpecialized);
  FreeAligned(insideUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(insideGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(containsRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(containsCuda);
  FreeAligned(insideCuda);
#endif
  return mismatches + insidemismatches;
}

int Benchmarker::CompareMetaInformation() const
{

  double vecgeomcapacity = 0.;
  for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
    vecgeomcapacity += (const_cast<VPlacedVolume *>(v->Specialized()))->Capacity();
    printf("## VecGeom capacity sum %lf\n", vecgeomcapacity);
  }
#ifdef VECGEOM_ROOT
  if (fOkToRunROOT) {
    double ROOTcapacity = 0.;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      ROOTcapacity += v->ROOT()->Capacity();
    }
    printf("## ROOT capacity sum %lf\n", ROOTcapacity);
  }
#endif

#ifdef VECGEOM_GEANT4
  if (fOkToRunG4) {
    double g4capacity = 0.;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      g4capacity += const_cast<G4VSolid *>(v->Geant4())->GetCubicVolume();
    }
    printf("## G4 capacity sum %lf\n", g4capacity);
  }
#endif
  return 0;
}

int Benchmarker::RunToInBenchmark()
{
  VECGEOM_ASSERT(fWorld);

  if (fVerbosity > 0) {
    printf("Running DistanceToIn and SafetyToIn benchmark for %i points for "
           "%i repetitions.\n",
           fPointCount, fRepetitions);
  }

  // Allocate memory
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  if (fStepMax) FreeAligned(fStepMax);

  fPointPool     = new SOA3D<Precision>(fPointCount * fPoolMultiplier);
  fDirectionPool = new SOA3D<Precision>(fPointCount * fPoolMultiplier);
  fStepMax       = AllocateAligned<Precision>();
  for (unsigned i = 0; i < fPointCount; ++i)
    fStepMax[i] = kInfLength;

  if (fVerbosity > 1) printf("Generating points with bias %f...", fToInBias);

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  Stopwatch timer;
  timer.Start();
  volumeUtilities::FillUncontainedPoints(*fWorld, *fPointPool, fBenchmarkTop);
  volumeUtilities::FillBiasedDirections(*fWorld, *fPointPool, fToInBias, *fDirectionPool, fBenchmarkTop);
  if (fVerbosity > 1) printf(" Done in %f s.\n", timer.Stop());

  fPointPool->resize(fPointCount * fPoolMultiplier);
  fDirectionPool->resize(fPointCount * fPoolMultiplier);

  std::stringstream outputLabels;
  outputLabels << "Specialized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized   = AllocateAligned<Precision>();
  Precision *const safetiesSpecialized    = AllocateAligned<Precision>();
  Precision *const distancesUnspecialized = AllocateAligned<Precision>();
  Precision *const safetiesUnspecialized  = AllocateAligned<Precision>();
#ifdef VECGEOM_GEANT4
  Precision *const distancesGeant4 = AllocateAligned<Precision>();
  Precision *const safetiesGeant4  = AllocateAligned<Precision>();
  outputLabels << " - Geant4";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateAligned<Precision>();
  Precision *const safetiesRoot  = AllocateAligned<Precision>();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  Precision *const distancesCuda = AllocateAligned<Precision>();
  Precision *const safetiesCuda  = AllocateAligned<Precision>();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
#ifndef VECCORE_CUDA
  enable_counters_ARGH();
#endif
  for (unsigned int i = 0; i < fMeasurementCount; ++i) {
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToInBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunToInBenchmark, __itt_null, __itt_null, __itt_RunToInSpecialized);
#endif
    RunToInSpecialized(distancesSpecialized, safetiesSpecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToInBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunToInBenchmark, __itt_null, __itt_null, __itt_RunToInUnspecialized);
#endif
    RunToInUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToInBenchmark);
#endif
#ifdef VECGEOM_GEANT4
    if (fOkToRunG4) RunToInGeant4(distancesGeant4, safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
    if (fOkToRunROOT) RunToInRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
    RunToInCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), fDirectionPool->x(), fDirectionPool->y(),
                fDirectionPool->z(), distancesCuda, safetiesCuda);
#endif
  }
#ifndef VECCORE_CUDA
  disable_counters_ARGH();
#endif
  int errorcode = CompareDistances(fPointPool, fDirectionPool, distancesSpecialized, distancesUnspecialized,
#ifdef VECGEOM_ROOT
                                   distancesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                   distancesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                   distancesCuda,
#endif
                                   "DistanceToIn");

  // Clean up memory
  FreeAligned(distancesSpecialized);
  FreeAligned(distancesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(distancesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(distancesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(distancesCuda);
#endif

  // for the moment; do not consider safety for errorcodes
  // errorcode += CompareSafeties(
  CompareSafeties(fPointPool, NULL, safetiesSpecialized, safetiesUnspecialized,
#ifdef VECGEOM_ROOT
                  safetiesRoot,
#endif
#ifdef VECGEOM_GEANT4
                  safetiesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                  safetiesCuda,
#endif
                  "SafetyToIn");

  FreeAligned(safetiesSpecialized);
  FreeAligned(safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(safetiesCuda);
#endif
  return (errorcode) ? 1 : 0;
}

void Benchmarker::InitInsideCaches()
{
  Stopwatch timer;
  timer.Start();
  // initializes a reusable container of tracks inside the test volumes
  fInsidePointPoolCache     = new SOA3D<Precision>(fPointCount * fPoolMultiplier);
  fInsideDirectionPoolCache = new SOA3D<Precision>(fPointCount * fPoolMultiplier);
  volumeUtilities::FillContainedPoints(*fWorld, *fInsidePointPoolCache, false, fBenchmarkTop);
  volumeUtilities::FillRandomDirections(*fInsideDirectionPoolCache);
  fInsideCacheInitialized = true;
  if (fVerbosity > 1) printf(" Inside Caches initialized in %f s.\n", timer.Stop());
}

int Benchmarker::RunToOutBenchmark()
{

  VECGEOM_ASSERT(fWorld);

  if (fVerbosity > 0) {
    printf("Running DistanceToOut and SafetyToOut benchmark for %i points for "
           "%i repetitions.\n",
           fPointCount, fRepetitions);
  }
  fStepMax = AllocateAligned<Precision>();
  for (unsigned i = 0; i < fPointCount; ++i)
    fStepMax[i] = kInfLength;

  if (fVerbosity > 1) printf("Generating points...");

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  if (!fInsideCacheInitialized) {
    InitInsideCaches();
  }
  // copy from cache
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  fPointPool     = new SOA3D<Precision>(*fInsidePointPoolCache);
  fDirectionPool = new SOA3D<Precision>(*fInsideDirectionPoolCache);

  if (fVerbosity > 1) printf(" Done.\n");

  fPointPool->resize(fPointCount * fPoolMultiplier);
  fDirectionPool->resize(fPointCount * fPoolMultiplier);

  std::stringstream outputLabels;
  outputLabels << "Specialized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized   = AllocateAligned<Precision>();
  Precision *const safetiesSpecialized    = AllocateAligned<Precision>();
  Precision *const distancesUnspecialized = AllocateAligned<Precision>();
  Precision *const safetiesUnspecialized  = AllocateAligned<Precision>();
#ifdef VECGEOM_GEANT4
  Precision *const distancesGeant4 = AllocateAligned<Precision>();
  Precision *const safetiesGeant4  = AllocateAligned<Precision>();
  outputLabels << " - Geant4";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateAligned<Precision>();
  Precision *const safetiesRoot  = AllocateAligned<Precision>();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  Precision *const distancesCuda = AllocateAligned<Precision>();
  Precision *const safetiesCuda  = AllocateAligned<Precision>();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
#ifndef VECCORE_CUDA
  enable_counters_ARGH();
#endif
  for (unsigned int i = 0; i < fMeasurementCount; ++i) {
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToOutBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunToOutBenchmark, __itt_null, __itt_null, __itt_RunToOutSpecialized);
#endif
    RunToOutSpecialized(distancesSpecialized, safetiesSpecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToOutBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunToOutBenchmark, __itt_null, __itt_null, __itt_RunToOutUnspecialized);
#endif
    RunToOutUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToOutBenchmark);
#endif
#ifdef VECGEOM_GEANT4
    if (fOkToRunG4) RunToOutGeant4(distancesGeant4, safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
    if (fOkToRunROOT) RunToOutRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
    RunToOutCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), fDirectionPool->x(), fDirectionPool->y(),
                 fDirectionPool->z(), distancesCuda, safetiesCuda);
#endif
  }
#ifndef VECCORE_CUDA
  disable_counters_ARGH();
#endif

  int errorcode = CompareDistances(fPointPool, fDirectionPool, distancesSpecialized, distancesUnspecialized,
#ifdef VECGEOM_ROOT
                                   distancesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                   distancesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                   distancesCuda,
#endif
                                   "DistanceToOut");

  // Clean up memory
  FreeAligned(distancesSpecialized);
  FreeAligned(distancesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(distancesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(distancesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(distancesCuda);
#endif

  // errorcode += CompareSafeties(
  CompareSafeties(fPointPool, NULL, safetiesSpecialized, safetiesUnspecialized,
#ifdef VECGEOM_ROOT
                  safetiesRoot,
#endif
#ifdef VECGEOM_GEANT4
                  safetiesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                  safetiesCuda,
#endif
                  "SafetyToOut");

  FreeAligned(safetiesSpecialized);
  FreeAligned(safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(safetiesCuda);
#endif
  return (errorcode) ? 1 : 0;
}

int Benchmarker::RunToOutFromBoundaryBenchmark()
{

  VECGEOM_ASSERT(fWorld);

  if (fVerbosity > 0) {
    printf("Running DistanceToOutFromBoundary and SafetyToOutOnBoundary "
           "benchmark for %i points for "
           "%i repetitions.\n",
           fPointCount, fRepetitions);
  }

  fStepMax = AllocateAligned<Precision>();
  for (unsigned i = 0; i < fPointCount; ++i)
    fStepMax[i] = kInfLength;

  if (fVerbosity > 1) printf("Generating points ON BOUNDARY...");

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  if (!fInsideCacheInitialized) {
    InitInsideCaches();
  }
  // copy from cache
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  fPointPool     = new SOA3D<Precision>(*fInsidePointPoolCache);
  fDirectionPool = new SOA3D<Precision>(*fInsideDirectionPoolCache);

  // fetch the actual test placed volume
  if (fVolumes.size() > 1) {
    printf("WARNING: this test is currently only implemented for 1 testing "
           "volume\n");
  }
  auto targetvolume = fVolumes.front().Specialized();

  // now transport tracks to boundary and invert direction
  // with this procedure we make sure that we expect a valid
  // intersection when inverting the direction
  for (size_t track = 0; track < fPointPool->size(); ++track) {
    auto currentpoint = fPointPool->operator[](track);
    // make sure the generated points are in the (unplaced) reference frame of
    // targetvolume
    VECGEOM_ASSERT(targetvolume->UnplacedContains(currentpoint));

    auto dir           = fDirectionPool->operator[](track);
    auto boundarypoint = currentpoint + dir * targetvolume->DistanceToOut(currentpoint, dir);
    // make sure that new generated point is on the boundary of targetvolume
    // VECGEOM_ASSERT(targetvolume->UnplacedInside(boundarypoint) == vecgeom::kSurface);
    fPointPool->set(track, boundarypoint);
    fDirectionPool->set(track, -dir);
  }

  if (fVerbosity > 1) printf(" Done.\n");

  fPointPool->resize(fPointCount * fPoolMultiplier);
  fDirectionPool->resize(fPointCount * fPoolMultiplier);

  std::stringstream outputLabels;
  outputLabels << "Specialized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized   = AllocateAligned<Precision>();
  Precision *const safetiesSpecialized    = AllocateAligned<Precision>();
  Precision *const distancesUnspecialized = AllocateAligned<Precision>();
  Precision *const safetiesUnspecialized  = AllocateAligned<Precision>();
#ifdef VECGEOM_GEANT4
  Precision *const distancesGeant4 = AllocateAligned<Precision>();
  Precision *const safetiesGeant4  = AllocateAligned<Precision>();
  outputLabels << " - Geant4";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateAligned<Precision>();
  Precision *const safetiesRoot  = AllocateAligned<Precision>();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  Precision *const distancesCuda = AllocateAligned<Precision>();
  Precision *const safetiesCuda  = AllocateAligned<Precision>();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
  for (unsigned int i = 0; i < fMeasurementCount; ++i) {
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToOutBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunToOutBenchmark, __itt_null, __itt_null, __itt_RunToOutSpecialized);
#endif
    RunToOutSpecialized(distancesSpecialized, safetiesSpecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToOutBenchmark);
#endif
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_begin(__itt_RunToOutBenchmark, __itt_null, __itt_null, __itt_RunToOutUnspecialized);
#endif
    RunToOutUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#if defined(VECGEOM_TEST_VTUNE)
    __itt_task_end(__itt_RunToOutBenchmark);
#endif
#ifdef VECGEOM_GEANT4
    if (fOkToRunG4) RunToOutGeant4(distancesGeant4, safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
    if (fOkToRunROOT) RunToOutRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
    RunToOutCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), fDirectionPool->x(), fDirectionPool->y(),
                 fDirectionPool->z(), distancesCuda, safetiesCuda);
#endif
  }

  int errorcode = CompareDistances(fPointPool, fDirectionPool, distancesSpecialized, distancesUnspecialized,
#ifdef VECGEOM_ROOT
                                   distancesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                   distancesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                   distancesCuda,
#endif
                                   "DistanceToOutFromBoundary");

  // Clean up memory
  FreeAligned(distancesSpecialized);
  FreeAligned(distancesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(distancesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(distancesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(distancesCuda);
#endif

  errorcode += CheckSafetiesOnBoundary(fPointPool, NULL, safetiesSpecialized, safetiesUnspecialized,
#ifdef VECGEOM_ROOT
                                       safetiesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                       safetiesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                       safetiesCuda,
#endif
                                       "SafetyToOutONBDR");

  FreeAligned(safetiesSpecialized);
  FreeAligned(safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(safetiesCuda);
#endif
  return (errorcode) ? 1 : 0;
}

int Benchmarker::RunToOutFromBoundaryExitingBenchmark()
{

  VECGEOM_ASSERT(fWorld);

  if (fVerbosity > 0) {
    printf("Running DistanceToOutFromBoundaryExiting and SafetyToOutOnBoundary "
           "benchmark for %i points for "
           "%i repetitions.\n",
           fPointCount, fRepetitions);
  }

  fStepMax = AllocateAligned<Precision>();
  for (unsigned i = 0; i < fPointCount; ++i)
    fStepMax[i] = kInfLength;

  if (fVerbosity > 1) printf("Generating points ON BOUNDARY...");

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  if (!fInsideCacheInitialized) {
    InitInsideCaches();
  }
  // copy from cache
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  fPointPool     = new SOA3D<Precision>(*fInsidePointPoolCache);
  fDirectionPool = new SOA3D<Precision>(*fInsideDirectionPoolCache);

  // fetch the actual test placed volume
  if (fVolumes.size() > 1) {
    printf("WARNING: this test is currently only implemented for 1 testing "
           "volume\n");
  }
  auto targetvolume = fVolumes.front().Specialized();

  // now transport tracks to boundary and invert direction
  // with this procedure we make sure that we expect a valid
  // intersection when inverting the direction
  for (size_t track = 0; track < fPointPool->size(); ++track) {
    auto currentpoint = fPointPool->operator[](track);
    // make sure the generated points are in the (unplaced) reference frame of
    // targetvolume
    VECGEOM_ASSERT(targetvolume->UnplacedContains(currentpoint));

    auto dir           = fDirectionPool->operator[](track);
    auto boundarypoint = currentpoint + dir * targetvolume->DistanceToOut(currentpoint, dir);
    // make sure that new generated point is on the boundary of targetvolume
    // VECGEOM_ASSERT(targetvolume->UnplacedInside(boundarypoint) == vecgeom::kSurface);
    fPointPool->set(track, boundarypoint);
  }

  if (fVerbosity > 1) printf(" Done.\n");

  fPointPool->resize(fPointCount * fPoolMultiplier);
  fDirectionPool->resize(fPointCount * fPoolMultiplier);

  std::stringstream outputLabels;
  outputLabels << "Specialized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized   = AllocateAligned<Precision>();
  Precision *const safetiesSpecialized    = AllocateAligned<Precision>();
  Precision *const distancesUnspecialized = AllocateAligned<Precision>();
  Precision *const safetiesUnspecialized  = AllocateAligned<Precision>();
#ifdef VECGEOM_GEANT4
  Precision *const distancesGeant4 = AllocateAligned<Precision>();
  Precision *const safetiesGeant4  = AllocateAligned<Precision>();
  outputLabels << " - Geant4";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateAligned<Precision>();
  Precision *const safetiesRoot  = AllocateAligned<Precision>();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  Precision *const distancesCuda = AllocateAligned<Precision>();
  Precision *const safetiesCuda  = AllocateAligned<Precision>();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
  for (unsigned int i = 0; i < fMeasurementCount; ++i) {
    RunToOutSpecialized(distancesSpecialized, safetiesSpecialized);
    RunToOutUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
    if (fOkToRunG4) RunToOutGeant4(distancesGeant4, safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
    if (fOkToRunROOT) RunToOutRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
    RunToOutCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), fDirectionPool->x(), fDirectionPool->y(),
                 fDirectionPool->z(), distancesCuda, safetiesCuda);
#endif
  }

  int errorcode = CompareDistances(fPointPool, fDirectionPool, distancesSpecialized, distancesUnspecialized,
#ifdef VECGEOM_ROOT
                                   distancesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                   distancesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                   distancesCuda,
#endif
                                   "DistanceToOutFromBoundaryExiting");

  // Clean up memory
  FreeAligned(distancesSpecialized);
  FreeAligned(distancesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(distancesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(distancesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(distancesCuda);
#endif

  errorcode += CheckSafetiesOnBoundary(fPointPool, NULL, safetiesSpecialized, safetiesUnspecialized,
#ifdef VECGEOM_ROOT
                                       safetiesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                       safetiesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                       safetiesCuda,
#endif
                                       "SafetyToOutONBDR");

  FreeAligned(safetiesSpecialized);
  FreeAligned(safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(safetiesCuda);
#endif
  return (errorcode) ? 1 : 0;
}

int Benchmarker::RunToInFromBoundaryBenchmark()
{

  VECGEOM_ASSERT(fWorld);

  if (fVerbosity > 0) {
    printf("Running DistanceToInFromBoundary and SafetyToInFromBoundary "
           "benchmark for %i points for "
           "%i repetitions.\n",
           fPointCount, fRepetitions);
  }

  if (fVerbosity > 1) printf("Generating points ON BOUNDARY...");

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  if (!fInsideCacheInitialized) {
    InitInsideCaches();
  }
  // copy from cache
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  fPointPool     = new SOA3D<Precision>(*fInsidePointPoolCache);
  fDirectionPool = new SOA3D<Precision>(*fInsideDirectionPoolCache);

  // fetch the actual test placed volume
  if (fVolumes.size() > 1) {
    printf("WARNING: this test is currently only implemented for 1 testing "
           "volume\n");
  }
  auto targetvolume = fVolumes.front().Specialized();

  // now transport tracks to boundary and invert direction
  // with this procedure we make sure that we expect a valid
  // intersection when inverting the direction
  for (size_t track = 0; track < fPointPool->size(); ++track) {
    auto currentpoint = fPointPool->operator[](track);
    // make sure the generated points are in the (unplaced) reference frame of
    // targetvolume
    VECGEOM_ASSERT(targetvolume->UnplacedContains(currentpoint));

    auto dir           = fDirectionPool->operator[](track);
    auto boundarypoint = currentpoint + dir * targetvolume->DistanceToOut(currentpoint, dir);
    // make sure that new generated point is on the boundary of targetvolume
    // VECGEOM_ASSERT(targetvolume->UnplacedInside(boundarypoint) == vecgeom::kSurface);
    fPointPool->set(track, boundarypoint);
    fDirectionPool->set(track, -dir);
  }

  if (fVerbosity > 1) printf(" Done.\n");

  fPointPool->resize(fPointCount * fPoolMultiplier);
  fDirectionPool->resize(fPointCount * fPoolMultiplier);

  std::stringstream outputLabels;
  outputLabels << "Specialized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized   = AllocateAligned<Precision>();
  Precision *const safetiesSpecialized    = AllocateAligned<Precision>();
  Precision *const distancesUnspecialized = AllocateAligned<Precision>();
  Precision *const safetiesUnspecialized  = AllocateAligned<Precision>();
#ifdef VECGEOM_GEANT4
  Precision *const distancesGeant4 = AllocateAligned<Precision>();
  Precision *const safetiesGeant4  = AllocateAligned<Precision>();
  outputLabels << " - Geant4";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateAligned<Precision>();
  Precision *const safetiesRoot  = AllocateAligned<Precision>();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  Precision *const distancesCuda = AllocateAligned<Precision>();
  Precision *const safetiesCuda  = AllocateAligned<Precision>();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
  for (unsigned int i = 0; i < fMeasurementCount; ++i) {
    RunToInSpecialized(distancesSpecialized, safetiesSpecialized);
    RunToInUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
    if (fOkToRunG4) RunToInGeant4(distancesGeant4, safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
    if (fOkToRunROOT) RunToInRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
    RunToInCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), fDirectionPool->x(), fDirectionPool->y(),
                fDirectionPool->z(), distancesCuda, safetiesCuda);
#endif
  }

  int errorcode =
      CheckDistancesFromBoundary(0., fPointPool, fDirectionPool, distancesSpecialized, distancesUnspecialized,
#ifdef VECGEOM_ROOT
                                 distancesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                 distancesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                 distancesCuda,
#endif
                                 "DistanceToInFromBoundary");

  // Clean up memory
  FreeAligned(distancesSpecialized);
  FreeAligned(distancesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(distancesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(distancesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(distancesCuda);
#endif

  errorcode += CheckSafetiesOnBoundary(fPointPool, NULL, safetiesSpecialized, safetiesUnspecialized,
#ifdef VECGEOM_ROOT
                                       safetiesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                       safetiesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                       safetiesCuda,
#endif
                                       "SafetyToInONBDR");

  FreeAligned(safetiesSpecialized);
  FreeAligned(safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(safetiesCuda);
#endif
  return (errorcode) ? 1 : 0;
}

int Benchmarker::RunToInFromBoundaryExitingBenchmark()
{
  VECGEOM_ASSERT(fWorld);

  if (fVerbosity > 0) {
    printf("Running DistanceToInExitingFromBoundary and SafetyToInExitingFromBoundary "
           "benchmark for %i points for "
           "%i repetitions.\n",
           fPointCount, fRepetitions);
  }

  fStepMax = AllocateAligned<Precision>();
  for (unsigned i = 0; i < fPointCount; ++i)
    fStepMax[i] = kInfLength;

  if (fVerbosity > 1) printf("Generating points ON BOUNDARY...");

  // Generate points not contained in any daughters and set the fraction hitting
  // a daughter to the specified bias.
  if (!fInsideCacheInitialized) {
    InitInsideCaches();
  }
  // copy from cache
  if (fPointPool) delete fPointPool;
  if (fDirectionPool) delete fDirectionPool;
  fPointPool     = new SOA3D<Precision>(*fInsidePointPoolCache);
  fDirectionPool = new SOA3D<Precision>(*fInsideDirectionPoolCache);
  // fetch the actual test placed volume
  if (fVolumes.size() > 1) {
    printf("WARNING: this test is currently only implemented for 1 testing "
           "volume\n");
  }
  auto targetvolume = fVolumes.front().Specialized();

  // now transport tracks to boundary and invert direction
  // with this procedure we make sure that we expect a valid
  // intersection when inverting the direction
  for (size_t track = 0; track < fPointPool->size(); ++track) {
    auto currentpoint = fPointPool->operator[](track);
    // make sure the generated points are in the (unplaced) reference frame of
    // targetvolume
    VECGEOM_ASSERT(targetvolume->UnplacedContains(currentpoint));

    auto dir           = fDirectionPool->operator[](track);
    auto boundarypoint = currentpoint + dir * targetvolume->DistanceToOut(currentpoint, dir);
    // make sure that new generated point is on the boundary of targetvolume
    // VECGEOM_ASSERT(targetvolume->UnplacedInside(boundarypoint) == vecgeom::kSurface);
    fPointPool->set(track, boundarypoint);

    // do not change directions !
  }

  if (fVerbosity > 1) printf(" Done.\n");

  fPointPool->resize(fPointCount * fPoolMultiplier);
  fDirectionPool->resize(fPointCount * fPoolMultiplier);

  std::stringstream outputLabels;
  outputLabels << "Specialized - Unspecialized";

  // Allocate output memory
  Precision *const distancesSpecialized   = AllocateAligned<Precision>();
  Precision *const safetiesSpecialized    = AllocateAligned<Precision>();
  Precision *const distancesUnspecialized = AllocateAligned<Precision>();
  Precision *const safetiesUnspecialized  = AllocateAligned<Precision>();
#ifdef VECGEOM_GEANT4
  Precision *const distancesGeant4 = AllocateAligned<Precision>();
  Precision *const safetiesGeant4  = AllocateAligned<Precision>();
  outputLabels << " - Geant4";
#endif
#ifdef VECGEOM_ROOT
  Precision *const distancesRoot = AllocateAligned<Precision>();
  Precision *const safetiesRoot  = AllocateAligned<Precision>();
  outputLabels << " - ROOT";
#endif
#ifdef VECGEOM_ENABLE_CUDA
  Precision *const distancesCuda = AllocateAligned<Precision>();
  Precision *const safetiesCuda  = AllocateAligned<Precision>();
  outputLabels << " - CUDA";
#endif

  // Run all benchmarks
  for (unsigned int i = 0; i < fMeasurementCount; ++i) {
    RunToInSpecialized(distancesSpecialized, safetiesSpecialized);
    RunToInUnspecialized(distancesUnspecialized, safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
    if (fOkToRunG4) RunToInGeant4(distancesGeant4, safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
    if (fOkToRunROOT) RunToInRoot(distancesRoot, safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
    RunToInCuda(fPointPool->x(), fPointPool->y(), fPointPool->z(), fDirectionPool->x(), fDirectionPool->y(),
                fDirectionPool->z(), distancesCuda, safetiesCuda);
#endif
  }

  int errorcode = CompareDistances(fPointPool, fDirectionPool, distancesSpecialized, distancesUnspecialized,
#ifdef VECGEOM_ROOT
                                   distancesRoot,
#endif
#ifdef VECGEOM_GEANT4
                                   distancesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                                   distancesCuda,
#endif
                                   "DistanceToInFromBoundary");

  // Clean up memory
  FreeAligned(distancesSpecialized);
  FreeAligned(distancesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(distancesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(distancesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(distancesCuda);
#endif

  // errorcode += CompareSafeties(
  CheckSafetiesOnBoundary(fPointPool, NULL, safetiesSpecialized, safetiesUnspecialized,
#ifdef VECGEOM_ROOT
                          safetiesRoot,
#endif
#ifdef VECGEOM_GEANT4
                          safetiesGeant4,
#endif
#ifdef VECGEOM_ENABLE_CUDA
                          safetiesCuda,
#endif
                          "SafetyToInONBDR");

  FreeAligned(safetiesSpecialized);
  FreeAligned(safetiesUnspecialized);
#ifdef VECGEOM_GEANT4
  FreeAligned(safetiesGeant4);
#endif
#ifdef VECGEOM_ROOT
  FreeAligned(safetiesRoot);
#endif
#ifdef VECGEOM_ENABLE_CUDA
  FreeAligned(safetiesCuda);
#endif
  return (errorcode) ? 1 : 0;
}

void Benchmarker::RunInsideSpecialized(bool *contains, Inside_t *inside)
{
  if (fVerbosity > 0) printf("Specialized   - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        if (fBenchmarkTop)
          contains[i] = v->Specialized()->GetUnplacedVolume()->Contains((*fPointPool)[index + i]);
        else
          contains[i] = v->Specialized()->Contains((*fPointPool)[index + i]);
      }
    }
  }
  Precision elapsedContains = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        if (fBenchmarkTop)
          inside[i] = v->Specialized()->GetUnplacedVolume()->Inside((*fPointPool)[index + i]);
        else
          inside[i] = v->Specialized()->Inside((*fPointPool)[index + i]);
      }
    }
  }
  Precision elapsedInside = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("Inside: %.6fs (%.6fs), Contains: %.6fs (%.6fs), "
           "Inside/Contains: %.2f\n",
           elapsedInside, elapsedInside / fVolumes.size(), elapsedContains, elapsedContains / fVolumes.size(),
           elapsedInside / elapsedContains);
  }
  fResults.push_back(GenerateBenchmarkResult(elapsedContains, kBenchmarkContains, kBenchmarkSpecialized, fInsideBias));
  fResults.push_back(GenerateBenchmarkResult(elapsedInside, kBenchmarkInside, kBenchmarkSpecialized, fInsideBias));
}

void Benchmarker::RunToInSpecialized(Precision *distances, Precision *safeties)
{
  if (fVerbosity > 0) printf("Specialized   - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          distances[i] = v->Specialized()->GetUnplacedVolume()->DistanceToIn((*fPointPool)[p], (*fDirectionPool)[p]);
        else
          distances[i] = v->Specialized()->DistanceToIn((*fPointPool)[p], (*fDirectionPool)[p]);
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          safeties[i] = v->Specialized()->GetUnplacedVolume()->SafetyToIn((*fPointPool)[p]);
        else
          safeties[i] = v->Specialized()->SafetyToIn((*fPointPool)[p]);
      }
    }
  }
  Precision elapsedSafety = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToIn: %.6fs (%.6fs), SafetyToIn: %.6fs (%.6fs), "
           "DistanceToIn/SafetyToIn: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(
      GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkSpecialized, fToInBias));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkSpecialized, fToInBias));
}

void Benchmarker::RunToOutSpecialized(Precision *distances, Precision *safeties)
{
  if (fVerbosity > 0) printf("Specialized   - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          distances[i] = v->Specialized()->GetUnplacedVolume()->DistanceToOut((*fPointPool)[p], (*fDirectionPool)[p]);
        else
          distances[i] = v->Specialized()->DistanceToOut((*fPointPool)[p], (*fDirectionPool)[p]);
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          safeties[i] = v->Specialized()->GetUnplacedVolume()->SafetyToOut((*fPointPool)[p]);
        else
          safeties[i] = v->Specialized()->SafetyToOut((*fPointPool)[p]);
      }
    }
  }
  Precision elapsedSafety = timer.Stop();

  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToOut: %.6fs (%.6fs), SafetyToOut: %.6fs (%.6fs), "
           "DistanceToOut/SafetyToOut: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToOut, kBenchmarkSpecialized, 1));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToOut, kBenchmarkSpecialized, 1));
}

void Benchmarker::RunInsideUnspecialized(bool *contains, Inside_t *inside)
{
  if (fVerbosity > 0) printf("Unspecialized - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          contains[i] = v->Unspecialized()->GetUnplacedVolume()->Contains((*fPointPool)[p]);
        else
          contains[i] = v->Unspecialized()->Contains((*fPointPool)[p]);
      }
    }
  }
  Precision elapsedContains = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          inside[i] = v->Unspecialized()->GetUnplacedVolume()->Inside((*fPointPool)[p]);
        else
          inside[i] = v->Unspecialized()->Inside((*fPointPool)[p]);
      }
    }
  }
  Precision elapsedInside = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("Inside: %.6fs (%.6fs), Contains: %.6fs (%.6fs), "
           "Inside/Contains: %.2f\n",
           elapsedInside, elapsedInside / fVolumes.size(), elapsedContains, elapsedContains / fVolumes.size(),
           elapsedInside / elapsedContains);
  }
  fResults.push_back(
      GenerateBenchmarkResult(elapsedContains, kBenchmarkContains, kBenchmarkUnspecialized, fInsideBias));
  fResults.push_back(GenerateBenchmarkResult(elapsedInside, kBenchmarkInside, kBenchmarkUnspecialized, fInsideBias));
}

void Benchmarker::RunToInUnspecialized(Precision *const distances, Precision *const safeties)
{
  if (fVerbosity > 0) printf("Unspecialized - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          distances[i] = v->Unspecialized()->GetUnplacedVolume()->DistanceToIn((*fPointPool)[p], (*fDirectionPool)[p]);
        else
          distances[i] = v->Unspecialized()->DistanceToIn((*fPointPool)[p], (*fDirectionPool)[p]);
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          safeties[i] = v->Unspecialized()->GetUnplacedVolume()->SafetyToIn((*fPointPool)[p]);
        else
          safeties[i] = v->Unspecialized()->SafetyToIn((*fPointPool)[p]);
      }
    }
  }
  Precision elapsedSafety = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToIn: %.6fs (%.6fs), SafetyToIn: %.6fs (%.6fs), "
           "DistanceToIn/SafetyToIn: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(
      GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkUnspecialized, fToInBias));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkUnspecialized, fToInBias));
}

void Benchmarker::RunToOutUnspecialized(Precision *const distances, Precision *const safeties)
{
  if (fVerbosity > 0) printf("Unspecialized - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          distances[i] = v->Unspecialized()->GetUnplacedVolume()->DistanceToOut((*fPointPool)[p], (*fDirectionPool)[p]);
        else
          distances[i] = v->Unspecialized()->DistanceToOut((*fPointPool)[p], (*fDirectionPool)[p]);
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        if (fBenchmarkTop)
          safeties[i] = v->Unspecialized()->GetUnplacedVolume()->SafetyToOut((*fPointPool)[p]);
        else
          safeties[i] = v->Unspecialized()->SafetyToOut((*fPointPool)[p]);
      }
    }
  }
  Precision elapsedSafety = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToOut: %.6fs (%.6fs), SafetyToOut: %.6fs (%.6fs), "
           "DistanceToOut/SafetyToOut: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToOut, kBenchmarkUnspecialized, 1));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToOut, kBenchmarkUnspecialized, 1));
}

#ifdef VECGEOM_GEANT4
void Benchmarker::RunInsideGeant4(::EInside *const inside)
{
  if (fVerbosity > 0) printf("Geant4        - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      Transformation3D const *transformation = v->Unspecialized()->GetTransformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        const Vector3D<Precision> point =
            fBenchmarkTop ? (*fPointPool)[p] : transformation->Transform((*fPointPool)[p]);
        inside[i] = v->Geant4()->Inside(G4ThreeVector(point[0], point[1], point[2]));
      }
    }
  }
  Precision elapsed = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("Inside: %.6fs (%.6fs), Contains: -.------s (-.------s), "
           "Inside/Contains: -.--\n",
           elapsed, elapsed / fVolumes.size());
  }
  fResults.push_back(GenerateBenchmarkResult(elapsed, kBenchmarkInside, kBenchmarkGeant4, fInsideBias));
}
void Benchmarker::RunToInGeant4(Precision *distances, Precision *safeties)
{
  if (fVerbosity > 0) printf("Geant4        - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      Transformation3D const *transformation = v->Unspecialized()->GetTransformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        const Vector3D<Precision> point =
            fBenchmarkTop ? (*fPointPool)[p] : transformation->Transform((*fPointPool)[p]);
        const Vector3D<Precision> dir =
            fBenchmarkTop ? (*fDirectionPool)[p] : transformation->TransformDirection((*fDirectionPool)[p]);
        // printf("Geant4 RunToIn will get point %.6f %.6f
        // %.6f",point[0],point[1],point[2]);
        // printf("Geant4 RunToIn before transform point %.6f %.6f
        // %.6f",(*fPointPool)[p][0],(*fPointPool)[p][1],(*fPointPool)[p][2]);
        distances[i] = v->Geant4()->DistanceToIn(G4ThreeVector(point[0], point[1], point[2]),
                                                 G4ThreeVector(dir[0], dir[1], dir[2]));
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      Transformation3D const *transformation = v->Unspecialized()->GetTransformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        const Vector3D<Precision> point =
            fBenchmarkTop ? (*fPointPool)[p] : transformation->Transform((*fPointPool)[p]);
        safeties[i] = v->Geant4()->DistanceToIn(G4ThreeVector(point[0], point[1], point[2]));
      }
    }
  }
  Precision elapsedSafety = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToIn: %.6fs (%.6fs), SafetyToIn: %.6fs (%.6fs), "
           "DistanceToIn/SafetyToIn: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkGeant4, fToInBias));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkGeant4, fToInBias));
}

void Benchmarker::RunToOutGeant4(Precision *distances, Precision *safeties)
{
  if (fVerbosity > 0) printf("Geant4        - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p                           = index + i;
        const Vector3D<Precision> point = (*fPointPool)[p];
        const Vector3D<Precision> dir   = (*fDirectionPool)[p];
        distances[i] = v->Geant4()->DistanceToOut(G4ThreeVector(point[0], point[1], point[2]),
                                                  G4ThreeVector(dir[0], dir[1], dir[2]), false, NULL, NULL);
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p                           = index + i;
        const Vector3D<Precision> point = (*fPointPool)[p];
        safeties[i]                     = v->Geant4()->DistanceToOut(G4ThreeVector(point[0], point[1], point[2]));
      }
    }
  }
  Precision elapsedSafety = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToOut: %.6fs (%.6fs), SafetyToOut: %.6fs (%.6fs), "
           "DistanceToOut/SafetyToOut: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToOut, kBenchmarkGeant4, 1));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToOut, kBenchmarkGeant4, 1));
}
#endif

#ifdef VECGEOM_ROOT
void Benchmarker::RunInsideRoot(bool *inside)
{
  if (fVerbosity > 0) printf("ROOT          - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      Transformation3D const *transformation = v->Unspecialized()->GetTransformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        const Vector3D<Precision> point =
            fBenchmarkTop ? (*fPointPool)[p] : transformation->Transform((*fPointPool)[p]);
        inside[i] = v->ROOT()->Contains(&point[0]);
      }
    }
  }
  Precision elapsed = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("Inside: -.------s (-.------s), Contains: %.6fs (%.6fs), "
           "Inside/Contains: -.--\n",
           elapsed, elapsed / fVolumes.size());
  }
  fResults.push_back(GenerateBenchmarkResult(elapsed, kBenchmarkContains, kBenchmarkRoot, fInsideBias));
}

void Benchmarker::RunToInRoot(Precision *const distances, Precision *const safeties)
{
  if (fVerbosity > 0) printf("ROOT          - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      Transformation3D const *transformation = v->Unspecialized()->GetTransformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        const Vector3D<Precision> point =
            fBenchmarkTop ? (*fPointPool)[p] : transformation->Transform((*fPointPool)[p]);
        const Vector3D<Precision> dir =
            fBenchmarkTop ? (*fDirectionPool)[p] : transformation->TransformDirection((*fDirectionPool)[p]);
        distances[i] = v->ROOT()->DistFromOutside(&point[0], &dir[0], 3);
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      Transformation3D const *transformation = v->Unspecialized()->GetTransformation();
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p = index + i;
        const Vector3D<Precision> point =
            fBenchmarkTop ? (*fPointPool)[p] : transformation->Transform((*fPointPool)[p]);
        safeties[i] = v->ROOT()->Safety(&point[0], false);
      }
    }
  }
  Precision elapsedSafety = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToIn: %.6fs (%.6fs), SafetyToIn: %.6fs (%.6fs), "
           "DistanceToIn/SafetyToIn: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkRoot, fToInBias));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkRoot, fToInBias));
}
void Benchmarker::RunToOutRoot(Precision *const distances, Precision *const safeties)
{
  if (fVerbosity > 0) printf("ROOT          - ");
  Stopwatch timer;
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p                  = index + i;
        Vector3D<double> point = (*fPointPool)[p];
        Vector3D<double> dir   = (*fDirectionPool)[p];
        distances[i]           = v->ROOT()->DistFromInside(&point[0], &dir[0], 3);
      }
    }
  }
  Precision elapsedDistance = timer.Stop();
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    int index = (rand() % fPoolMultiplier) * fPointCount;
    for (auto v = fVolumes.begin(), vEnd = fVolumes.end(); v != vEnd; ++v) {
      for (unsigned i = 0; i < fPointCount; ++i) {
        int p                  = index + i;
        Vector3D<double> point = (*fPointPool)[p];
        safeties[i]            = v->ROOT()->Safety(&point[0], true);
      }
    }
  }
  Precision elapsedSafety = timer.Stop();
  if (fVerbosity > 0 && fMeasurementCount == 1) {
    printf("DistanceToOut: %.6fs (%.6fs), SafetyToOut: %.6fs (%.6fs), "
           "DistanceToOut/SafetyToOut: %.2f\n",
           elapsedDistance, elapsedDistance / fVolumes.size(), elapsedSafety, elapsedSafety / fVolumes.size(),
           elapsedDistance / elapsedSafety);
  }
  fResults.push_back(GenerateBenchmarkResult(elapsedDistance, kBenchmarkDistanceToOut, kBenchmarkRoot, 1));
  fResults.push_back(GenerateBenchmarkResult(elapsedSafety, kBenchmarkSafetyToOut, kBenchmarkRoot, 1));
}
#endif

template <typename Type>
Type *Benchmarker::AllocateAligned() const
{
  return (Type *)vecCore::AlignedAlloc(kAlignmentBoundary, fPointCount * sizeof(Type));
}

template <typename Type>
void Benchmarker::FreeAligned(Type *const distance)
{
  if (distance) {
    vecCore::AlignedFree(distance);
  }
}

#ifdef VECGEOM_CUDA_INTERFACE
void Benchmarker::GetVolumePointers(std::list<DevicePtr<cuda::VPlacedVolume>> &volumesGpu)
{
  VECGEOM_DEVICE_API_CALL(DeviceSetLimit(VECGEOM_DEVICE_API_SYMBOL(LimitStackSize), 8192));
  CudaManager::Instance().LoadGeometry(GetWorld());
  CudaManager::Instance().Synchronize();
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin(); v != fVolumes.end(); ++v) {
    volumesGpu.push_back(CudaManager::Instance().LookupPlaced(v->Specialized()));
  }
}
#endif

} // End namespace vecgeom
