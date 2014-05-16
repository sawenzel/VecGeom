/// @file Benchmarker.h
/// @author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BENCHMARKING_BENCHMARKER_H_
#define VECGEOM_BENCHMARKING_BENCHMARKER_H_


#include "base/global.h"

#include "base/soa3d.h"
#include "benchmarking/BenchmarkResult.h"
#include "management/volume_pointers.h"
#include "volumes/placed_volume.h"

#include <list>

namespace vecgeom {

class Benchmarker {

private:

  VPlacedVolume const *fWorld;
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

public:

  void RunBenchmark();
  void RunInsideBenchmark();
  void RunToInBenchmark();
  void RunToOutBenchmark();
  
  Benchmarker() {}

  Benchmarker(VPlacedVolume const *const world);

  ~Benchmarker();

  unsigned GetPointCount() const { return fPointCount; }
  double GetToInBias() const { return fToInBias; }
  double GetInsideBias() const { return fInsideBias; }
  unsigned GetPoolMultiplier() const { return fPoolMultiplier; }
  int GetVerbosity() const { return fVerbosity; }
  unsigned GetRepetitions() const { return fRepetitions; }
  VPlacedVolume const* GetWorld() const { return fWorld; }

  void SetPointCount(const unsigned pointCount) { fPointCount = pointCount; }
  void SetToInBias(const double toInBias) { fToInBias = toInBias; }
  void SetInsideBias(const double insideBias) { fInsideBias = insideBias; }
  void SetPoolMultiplier(const unsigned poolMultiplier);
  void SetVerbosity(const int verbosity) { fVerbosity = verbosity; }
  void SetRepetitions(const unsigned repetitions) {
    fRepetitions = repetitions;
  }
  void SetWorld(VPlacedVolume const *const world);

  std::list<BenchmarkResult> const& GetResults() const { return fResults; }

  std::list<BenchmarkResult> PopResults();

private:
    
  void GenerateVolumePointers(VPlacedVolume const *const vol);

  BenchmarkResult GenerateBenchmarkResult(const double elapsed,
                                          const EBenchmarkedMethod method,
                                          const EBenchmarkedLibrary library,
                                          const double bias) const;

  void RunInsideSpecialized(bool *const inside);
  void RunToInSpecialized(Precision *const distances,
                          Precision *const safeties);
  void RunToOutSpecialized(Precision *const distances,
                           Precision *const safeties);

  void RunInsideVectorized(bool *const inside);
  void RunToInVectorized(Precision *const distances, Precision *const safeties);
  void RunToOutVectorized(Precision *const distances,
                          Precision *const safeties);

  void RunInsideUnspecialized(bool *const inside);
  void RunToInUnspecialized(Precision *const distances,
                            Precision *const safeties);
  void RunToOutUnspecialized(Precision *const distances,
                             Precision *const safeties);

#ifdef VECGEOM_USOLIDS
  void RunInsideUSolids(bool *const inside);
  void RunToInUSolids(double *const distances, Precision *const safeties);
  void RunToOutUSolids(double *const distances, Precision *const safeties);
#endif
#ifdef VECGEOM_ROOT
  void RunInsideRoot(bool *const inside);
  void RunToInRoot(double *const distances, Precision *const safeties);
  void RunToOutRoot(double *const distances, Precision *const safeties);
#endif
#ifdef VECGEOM_CUDA
  void RunInsideCuda(
    Precision *const pos_x, Precision *const pos_y,
    Precision *const pos_z, bool *const inside);
  void RunToInCuda(
    Precision *const pos_x, Precision *const pos_y,
    Precision *const pos_z, Precision *const dir_x,
    Precision *const dir_y, Precision *const dir_z,
    Precision *const distances, Precision *const safeties);
  void RunToOutCuda(
    Precision *const pos_x, Precision *const pos_y,
    Precision *const pos_z, Precision *const dir_x,
    Precision *const dir_y, Precision *const dir_z,
    Precision *const distances, Precision *const safeties);
#endif

  template <typename Type>
  Type* AllocateAligned() const;

  template <typename Type>
  static void FreeAligned(Type *const distance);

  void CompareDistances(
    Precision const *const specialized,
    Precision const *const vectorized,
    Precision const *const unspecialized,
#ifdef VECGEOM_ROOT
    Precision const *const root,
#endif
#ifdef VECGEOM_USOLIDS
    Precision const *const usolids,
#endif
#ifdef VECGEOM_CUDA
    Precision const *const cuda,
#endif
    char const *const method) const;

};

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_BENCHMARKER_H_