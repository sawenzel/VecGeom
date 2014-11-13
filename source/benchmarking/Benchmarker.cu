/// \file Benchmarker.cu
/// \author Johannes de Fine Licht

#include "benchmarking/Benchmarker.h"

#include "base/Stopwatch.h"
#include "backend/cuda/Backend.h"
#include "management/CudaManager.h"

namespace vecgeom_cuda {

__global__
void ContainsBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const int n,
    bool *const contains) {
  const int i = ThreadIndex();
  if (i >= n) return;
  contains[i] = volume->Contains(positions[i]);
}

__global__
void InsideBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const int n,
    Inside_t *const inside) {
  const int i = ThreadIndex();
  if (i >= n) return;
  inside[i] = volume->Inside(positions[i]);
}

__global__
void DistanceToInBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const SOA3D<Precision> directions,
    const int n,
    Precision *const distance) {
  const int i = ThreadIndex();
  if (i >= n) return;
  distance[i] = volume->DistanceToIn(positions[i], directions[i]);
}

__global__
void DistanceToOutBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const SOA3D<Precision> directions,
    const int n,
    Precision *const distance) {
  const int i = ThreadIndex();
  if (i >= n) return;
  distance[i] = volume->DistanceToOut(positions[i], directions[i]);
}

__global__
void SafetyToInBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const int n,
    Precision *const distance) {
  const int i = ThreadIndex();
  if (i >= n) return;
  distance[i] = volume->SafetyToIn(positions[i]);
}

__global__
void SafetyToOutBenchmarkCudaKernel(
    VPlacedVolume const *const volume,
    const SOA3D<Precision> positions,
    const int n,
    Precision *const distance) {
  const int i = ThreadIndex();
  if (i >= n) return;
  distance[i] = volume->SafetyToOut(positions[i]);
}

} // End namespace vecgeom_cuda

namespace vecgeom {

void Benchmarker::RunInsideCuda(
    Precision *const posX, Precision *const posY,
    Precision *const posZ, bool *const contains, Inside_t *const inside) {

  typedef vecgeom_cuda::VPlacedVolume const* CudaVolume;
  typedef vecgeom_cuda::SOA3D<Precision> CudaSOA3D;

  if (fVerbosity > 0) printf("CUDA          - ");

  CudaManager::Instance().LoadGeometry(this->GetWorld());
  CudaManager::Instance().Synchronize();
  std::list<CudaVolume> volumesGpu;
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
       v != fVolumes.end(); ++v) {
    volumesGpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->Specialized())
      )
    );
  }

  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(fPointCount);

  vecgeom_cuda::Stopwatch timer;

  Precision *posXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));

  CudaSOA3D positionGpu  = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);

  bool *containsGpu =
      AllocateOnGpu<bool>(sizeof(bool)*fPointCount);
  Inside_t *insideGpu =
      AllocateOnGpu<Inside_t>(sizeof(Inside_t)*fPointCount);

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom_cuda::ContainsBenchmarkCudaKernel<<<launch.grid_size,
                                                  launch.block_size>>>(
        *v, positionGpu, fPointCount, containsGpu
      );
    }
  }
  Precision elapsedContains = timer.Stop();

  CopyFromGpu(containsGpu, contains, fPointCount*sizeof(bool));

  FreeFromGpu(containsGpu);

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom_cuda::InsideBenchmarkCudaKernel<<<launch.grid_size,
                                                launch.block_size>>>(
        *v, positionGpu, fPointCount, insideGpu
      );
    }
  }
  Precision elapsedInside = timer.Stop();

  CopyFromGpu(insideGpu, inside, fPointCount*sizeof(Inside_t));

  FreeFromGpu(insideGpu);
  FreeFromGpu(posXGpu);
  FreeFromGpu(posYGpu);
  FreeFromGpu(posZGpu);

  if (fVerbosity > 0) {
    printf("Inside: %.6fs (%.6fs), Contains: %.6fs (%.6fs), "
           "Inside/Contains: %.2f\n",
           elapsedInside, elapsedInside/fVolumes.size(),
           elapsedContains, elapsedContains/fVolumes.size(),
           elapsedInside/elapsedContains);
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedContains, kBenchmarkContains, kBenchmarkCuda, fInsideBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedInside, kBenchmarkInside, kBenchmarkCuda, fInsideBias
    )
  );

  cudaDeviceSynchronize();
}

void Benchmarker::RunToInCuda(
    Precision *const posX, Precision *const posY,
    Precision *const posZ, Precision *const dirX,
    Precision *const dirY, Precision *const dirZ,
    Precision *const distances, Precision *const safeties) {

  typedef vecgeom_cuda::VPlacedVolume const* CudaVolume;
  typedef vecgeom_cuda::SOA3D<Precision> CudaSOA3D;

  if (fVerbosity > 0) printf("CUDA          - ");

  CudaManager::Instance().LoadGeometry(this->GetWorld());
  CudaManager::Instance().Synchronize();
  std::list<CudaVolume> volumesGpu;
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
       v != fVolumes.end(); ++v) {
    volumesGpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->Specialized())
      )
    );
  }

  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(fPointCount);
  vecgeom_cuda::Stopwatch timer;

  Precision *posXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));
  CudaSOA3D positionGpu = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);

  Precision *dirXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(dirX, dirXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirY, dirYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirZ, dirZGpu, fPointCount*sizeof(Precision));
  CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, fPointCount);

  Precision *distancesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                     *fPointCount);
  Precision *safetiesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                    *fPointCount);

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom_cuda::DistanceToInBenchmarkCudaKernel<<<launch.grid_size,
                                                      launch.block_size>>>(
        *v, positionGpu, directionGpu, fPointCount, distancesGpu
      );
    }
  }
  Precision elapsedDistance = timer.Stop();

  CopyFromGpu(distancesGpu, distances, fPointCount*sizeof(Precision));

  FreeFromGpu(distancesGpu);
  FreeFromGpu(dirXGpu);
  FreeFromGpu(dirYGpu);
  FreeFromGpu(dirZGpu);

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom_cuda::SafetyToInBenchmarkCudaKernel<<<launch.grid_size,
                                                    launch.block_size>>>(
        *v, positionGpu, fPointCount, safetiesGpu
      );
    }
  }
  Precision elapsedSafety = timer.Stop();

  CopyFromGpu(safetiesGpu, safeties, fPointCount*sizeof(Precision));

  FreeFromGpu(safetiesGpu);
  FreeFromGpu(posXGpu);
  FreeFromGpu(posYGpu);
  FreeFromGpu(posZGpu);

  if (fVerbosity > 0) {
    printf("DistanceToIn: %.6fs (%.6fs), SafetyToIn: %.6fs (%.6fs), "
           "DistanceToIn/SafetyToIn: %.2f\n",
           elapsedDistance, elapsedDistance/fVolumes.size(),
           elapsedSafety, elapsedSafety/fVolumes.size(),
           elapsedDistance/elapsedSafety);
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistance, kBenchmarkDistanceToIn, kBenchmarkCuda, fToInBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafety, kBenchmarkSafetyToIn, kBenchmarkCuda, fToInBias
    )
  );

  cudaDeviceSynchronize();
}

void Benchmarker::RunToOutCuda(
    Precision *const posX, Precision *const posY,
    Precision *const posZ, Precision *const dirX,
    Precision *const dirY, Precision *const dirZ,
    Precision *const distances, Precision *const safeties) {

  typedef vecgeom_cuda::VPlacedVolume const* CudaVolume;
  typedef vecgeom_cuda::SOA3D<Precision> CudaSOA3D;

  double elapsedDistance;
  double elapsedSafety;

  if (fVerbosity > 0) printf("CUDA          - ");

  CudaManager::Instance().LoadGeometry(this->GetWorld());
  CudaManager::Instance().Synchronize();
  std::list<CudaVolume> volumesGpu;
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
       v != fVolumes.end(); ++v) {
    volumesGpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->Specialized())
      )
    );
  }

  Precision *posXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirX, dirXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirY, dirYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirZ, dirZGpu, fPointCount*sizeof(Precision));

  CudaSOA3D positionGpu  = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);
  CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, fPointCount);

  Precision *distancesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                     *fPointCount);
  Precision *safetiesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                    *fPointCount);

  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(fPointCount);
  vecgeom_cuda::Stopwatch timer;

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom_cuda::DistanceToOutBenchmarkCudaKernel<<<launch.grid_size,
                                                       launch.block_size>>>(
        *v, positionGpu, directionGpu, fPointCount, distancesGpu
      );
    }
  }
  elapsedDistance = timer.Stop();

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom_cuda::SafetyToOutBenchmarkCudaKernel<<<launch.grid_size,
                                                     launch.block_size>>>(
        *v, positionGpu, fPointCount, safetiesGpu
      );
    }
  }
  elapsedSafety = timer.Stop();

  if (fVerbosity > 0) {
    printf("DistanceToOut: %.6fs (%.6fs), SafetyToOut: %.6fs (%.6fs), "
           "DistanceToOut/SafetyToOut: %.2f\n",
           elapsedDistance, elapsedDistance/fVolumes.size(),
           elapsedSafety, elapsedSafety/fVolumes.size(),
           elapsedDistance/elapsedSafety);
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistance, kBenchmarkDistanceToOut, kBenchmarkCuda, 1
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafety, kBenchmarkSafetyToOut, kBenchmarkCuda, 1
    )
  );

  CopyFromGpu(distancesGpu, distances, fPointCount*sizeof(Precision));
  CopyFromGpu(safetiesGpu, safeties, fPointCount*sizeof(Precision));

  FreeFromGpu(distancesGpu);
  FreeFromGpu(safetiesGpu);
  FreeFromGpu(posXGpu);
  FreeFromGpu(posYGpu);
  FreeFromGpu(posZGpu);
  FreeFromGpu(dirXGpu);
  FreeFromGpu(dirYGpu);
  FreeFromGpu(dirZGpu);

  cudaDeviceSynchronize();
}

} // End namespace vecgeom
