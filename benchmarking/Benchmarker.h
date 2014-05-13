/**
 * @file ToInBenchmarker.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_BENCHMARKING_TOINBENCHMARKER_H_
#define VECGEOM_BENCHMARKING_TOINBENCHMARKER_H_


#include "base/global.h"

#include "base/soa3d.h"
#include "benchmarking/BenchmarkResult.h"
#include "management/volume_pointers.h"
#include "volumes/placed_volume.h"

#include <list>

namespace vecgeom {

class ToInBenchmarker {

private:

  VPlacedVolume const *fWorld;
  unsigned fPointCount;
  unsigned fPoolMultiplier;
  unsigned fRepetitions;
  std::list<VolumePointers> fVolumes;
  int fVerbose;
  double fBias;
  SOA3D<Precision> *fPointPool;
  SOA3D<Precision> *fDirectionPool;
  Precision *fStepMax;

public:

  void BenchmarkAll();
  void BenchmarkSpecialized();
  void BenchmarkUnspecialized();
  void BenchmarkVectorized();
#ifdef VECGEOM_USOLIDS
  void BenchmarkUSolids();
#endif
#ifdef VECGEOM_ROOT
  void BenchmarkRoot();
#endif
#ifdef VECGEOM_CUDA
  void BenchmarkCuda();
#endif
  
  ToInBenchmarker() {}

  ToInBenchmarker(VPlacedVolume const *const world);

  ~ToInBenchmarker();

  unsigned GetPointCount() const { return fPointCount; }
  double GetBias() const { return fBias; }
  unsigned GetPoolMultiplier() const { return fPoolMultiplier; }
  int GetVerbose() const { return fVerbose; }
  unsigned GetRepetitions() const { return fRepetitions; }
  VPlacedVolume const* GetWorld() const { return fWorld; }

  void SetPointCount(const unsigned pointCount) { fPointCount = pointCount; }
  void SetBias(const double bias) { fBias = bias; }
  void SetPoolMultiplier(const unsigned poolMultiplier);
  void SetVerbose(const int verbose) { fVerbose = verbose; }
  void SetRepetitions(const unsigned repetitions) {
    fRepetitions = repetitions;
  }
  void SetWorld(VPlacedVolume const *const world) { fWorld = world; }

private:
    
  void GenerateVolumePointers(VPlacedVolume const *const vol);

  BenchmarkResult GenerateBenchmarkResult(const double elapsed,
                                          const BenchmarkType type) const;

  void RunSpecialized(Precision *const distances,
                      Precision *const safeties) const;
  void RunVectorized(Precision *const distances,
                     Precision *const safeties) const;
  void RunUnspecialized(Precision *const distances,
                        Precision *const safeties) const;
#ifdef VECGEOM_USOLIDS
  void RunUSolids(double *const distances, Precision *const safeties) const;
#endif
#ifdef VECGEOM_ROOT
  void RunRoot(double *const distances, Precision *const safeties) const;
#endif
#ifdef VECGEOM_CUDA
  void RunCuda(
    Precision *const pos_x, Precision *const pos_y,
    Precision *const pos_z, Precision *const dir_x,
    Precision *const dir_y, Precision *const dir_z,
    Precision *const distances, Precision *const safeties) const;
#endif

  void PrepareBenchmark();

  double* AllocateDistance() const;

  static void FreeDistance(double *const distance);

};

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_TOINBENCHMARKER_H_