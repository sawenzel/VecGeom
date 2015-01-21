/// \file Benchmarker.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BENCHMARKING_BENCHMARKER_H_
#define VECGEOM_BENCHMARKING_BENCHMARKER_H_

#include "base/Global.h"

#include "volumes/PlacedVolume.h"
#include "base/SOA3D.h"
#include "benchmarking/BenchmarkResult.h"
#include "benchmarking/VolumePointers.h"

#ifdef VECGEOM_USOLIDS
#include "VUSolid.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4VSolid.hh"
#endif

#ifdef VECGEOM_CUDA
#include "backend/cuda/Interface.h"
#endif

#include <list>
#include <vector>

namespace vecgeom {

VECGEOM_HOST_FORWARD_DECLARE( class VPlacedVolume; );
VECGEOM_DEVICE_FORWARD_DECLARE( class VolumePointers; );

/// \brief Benchmarks geometry methods of arbitrary volumes for different
///        backends and compares to any included external libraries.
///
/// In order to run a benchmark, a world volume must be provided to the
/// benchmarker. This volume must have an available bounding box, and can
/// contain any number of daughter volumes. When the benchmark is run, points
/// will be sampled with a bias in regard to these daughter volumes. Deeper
/// hierarchies are not considered. For any level of verbosity above zero, the
/// benchmarker will output results to standard output. However, result structs
/// are available and can be retrieved, containing all information related to a
/// specific run.
/// \sa BenchmarkResult
class Benchmarker {

private:
  using VPlacedVolume_t = cxx::VPlacedVolume const *;

  VPlacedVolume_t fWorld;
  unsigned fPointCount;
  unsigned fPoolMultiplier;
  unsigned fRepetitions;
  std::list<VolumePointers> fVolumes;
  std::list<BenchmarkResult> fResults;
  int fVerbosity;
  double fToInBias, fInsideBias;
  SOA3D<Precision> *fPointPool;
  SOA3D<Precision> *fDirectionPool;
  Precision *fStepMax;

  // tolerance for comparisons
  Precision fTolerance;

  // containers to store problematic points
  // this can be filled during evaluation
  std::vector< Vector3D<Precision> > fProblematicContainPoints;

public:

  Benchmarker();

  /// \param world Mother volume containing daughters that will be benchmarked.
  ///              The mother volume must have an available bounding box, as it
  ///              is used in the sampling process.
  Benchmarker(VPlacedVolume_t const world);

  ~Benchmarker();

  /// \brief set tolerance for comparisons
  void SetTolerance(Precision tol) { fTolerance = tol; }

  /// \brief Runs all geometry benchmarks.
  /// return 0 if no error found; returns 1 if error found
  int RunBenchmark();


  /// \brief Runs some meta information functions (such as surface arrea, volume and so on) on registered shapes
  /// checks if this information agrees across different implementations (VecGeom, ROOT, Geant4, ...)
  int CompareMetaInformation() const;

  /// \brief Runs a benchmark of the Inside method.
  ///
  /// The fraction of sampled points that will be located inside of daughter
  /// volume is specified by calling SetInsideBias().
  /// \sa SetInsideBias(const double)
  /// return 0 if no error found; returns 1 if error found
  int RunInsideBenchmark();

  /// \brief Runs a benchmark of the DistanceToIn and SafetyToIn methods.
  ///
  /// The fraction of sampled points that should be hitting a daughter volume is
  /// specified by calling SetToInBias().
  /// \sa SetToInBias(const double)
  /// return 0 if no error found; returns 1 if error found
  int RunToInBenchmark();

  /// \brief Runs a benchmark of the DistanceToOut and SafetyToOut methods.
  /// return 0 if no error found; returns 1 if error found
  int RunToOutBenchmark();

  /// \return Amount of points and directions sampled for each benchmark
  ///         iteration.
  unsigned GetPointCount() const { return fPointCount; }

  /// \return Bias with which directions for DistanceToIn and SafetyToIn are
  ///         sampled.
  double GetToInBias() const { return fToInBias; }

  /// \return Bias with which the points for Inside are sampled.
  double GetInsideBias() const { return fInsideBias; }

  /// \return Multiplier of point and direction pool to use for simulating
  ///         random memory access.
  unsigned GetPoolMultiplier() const { return fPoolMultiplier; }

  /// \return Level of verbosity to standard output.
  int GetVerbosity() const { return fVerbosity; }

  /// \return Amount of iterations the benchmark is run for.
  unsigned GetRepetitions() const { return fRepetitions; }

  /// \return World whose daughters are benchmarked.
  VPlacedVolume_t GetWorld() const { return fWorld; }

  /// \param pointCount Amount of points to benchmark in each iteration.
  void SetPointCount(const unsigned pointCount) { fPointCount = pointCount; }

  /// \param toInBias Fraction of sampled particles that should be facing a
  /// daughter volume.
  void SetToInBias(const double toInBias) { fToInBias = toInBias; }

  /// \param insideBias Fraction of sampled particles that should be contained
  ///                   in a daughter volume.
  void SetInsideBias(const double insideBias) { fInsideBias = insideBias; }

  /// \param Multiplier for the pool of sampled points and directions.
  ///
  /// Can be increased to simulate more random access of memory, but will
  /// disable comparison of output.     
  void SetPoolMultiplier(const unsigned poolMultiplier);

  /// \param verbosity Regulates the amount of information printed to standard
  ///                  output.
  ///
  /// If set to zero nothing is printed, but results are stored and can be
  /// retrieved using the GetResults() or PopResults() methods.
  /// \sa GetResults()
  /// \sa PopResults()
  void SetVerbosity(const int verbosity) { fVerbosity = verbosity; }

  /// \param Amount of iterations to run the benchmarks.
  void SetRepetitions(const unsigned repetitions) {
    fRepetitions = repetitions;
  }

  /// \param World volume containing daughters to be benchmarked.
  void SetWorld(VPlacedVolume_t const world);

  /// \return List of results of previously performed benchmarks.
  std::list<BenchmarkResult> const& GetResults() const { return fResults; }

  /// \return List of results of previously performed benchmarks. Clears the
  ///         internal history.
  std::list<BenchmarkResult> PopResults();

  std::vector<Vector3D<Precision> > const & GetProblematicContainPoints() const {
      return fProblematicContainPoints;
  }

private:
    
  void GenerateVolumePointers(VPlacedVolume_t const vol);

  BenchmarkResult GenerateBenchmarkResult(const double elapsed,
                                          const EBenchmarkedMethod method,
                                          const EBenchmarkedLibrary library,
                                          const double bias) const;
  void RunInsideSpecialized(bool *contains, Inside_t *inside);
  void RunToInSpecialized(Precision *distances,
                          Precision *safeties);
  void RunToOutSpecialized(Precision *distances,
                           Precision *safeties);

  void RunInsideVectorized(bool *contains, Inside_t *inside);
  void RunToInVectorized(Precision *distances, Precision *safeties);
  void RunToOutVectorized(Precision *distances,
                          Precision *safeties);

  void RunInsideUnspecialized(bool *contains, Inside_t *inside);
  void RunToInUnspecialized(Precision *distances,
                            Precision *safeties);
  void RunToOutUnspecialized(Precision *distances,
                             Precision *safeties);

#ifdef VECGEOM_USOLIDS
  void RunInsideUSolids(::VUSolid::EnumInside *inside);
  void RunToInUSolids(double *distances, Precision *safeties);
  void RunToOutUSolids(double *distances, Precision *safeties);
#endif
#ifdef VECGEOM_ROOT
  void RunInsideRoot(bool *inside);
  void RunToInRoot(double *distances, Precision *safeties);
  void RunToOutRoot(double *distances, Precision *safeties);
#endif
#ifdef VECGEOM_GEANT4
  void RunInsideGeant4(::EInside *inside);
  void RunToInGeant4(double *distances, Precision *safeties);
  void RunToOutGeant4(double *distances, Precision *safeties);
#endif
#ifdef VECGEOM_CUDA
  void RunInsideCuda(
    Precision *posX, Precision *posY,
    Precision *posZ, bool *contains, Inside_t *inside);
  void RunToInCuda(
    Precision *posX, Precision *posY,
    Precision *posZ, Precision *dirX,
    Precision *dirY, Precision *dirZ,
    Precision *distances, Precision *safeties);
  void RunToOutCuda(
    Precision *posX, Precision *posY,
    Precision *posZ, Precision *dirX,
    Precision *dirY, Precision *dirZ,
    Precision *distances, Precision *safeties);
  void GetVolumePointers( std::list<cxx::DevicePtr<cuda::VPlacedVolume> > &volumesGpu );
#endif

  template <typename Type>
  Type* AllocateAligned() const;

  template <typename Type>
  static void FreeAligned(Type *const distance);

  int CompareDistances(
    SOA3D<Precision> *points,
    SOA3D<Precision> *directions,
    Precision const *const specialized,
    Precision const *const vectorized,
    Precision const *const unspecialized,
#ifdef VECGEOM_ROOT
    Precision const *const root,
#endif
#ifdef VECGEOM_USOLIDS
    Precision const *const usolids,
#endif
#ifdef VECGEOM_GEANT4
    Precision const *const geant4,
#endif
#ifdef VECGEOM_CUDA
    Precision const *const cuda,
#endif
    char const *const method) const;

  int CompareSafeties(
    SOA3D<Precision> *points,
    SOA3D<Precision> *directions,
    Precision const *const specialized,
    Precision const *const vectorized,
    Precision const *const unspecialized,
#ifdef VECGEOM_ROOT
    Precision const *const root,
#endif
#ifdef VECGEOM_USOLIDS
    Precision const *const usolids,
#endif
#ifdef VECGEOM_GEANT4
    Precision const *const geant4,
#endif
#ifdef VECGEOM_CUDA
    Precision const *const cuda,
#endif
    char const *const method) const;
  
};

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_BENCHMARKER_H_
