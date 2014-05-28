/// @file Benchmarker.cu
/// @author Johannes de Fine Licht

#include "benchmarking/Benchmarker.h"

#include "base/stopwatch.h"
#include "backend/cuda/backend.h"
#include "management/cuda_manager.h"

namespace vecgeom_cuda {

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
    Precision *const posZ, Inside_t *const inside) {

  typedef vecgeom_cuda::VPlacedVolume const* CudaVolume;
  typedef vecgeom_cuda::SOA3D<Precision> CudaSOA3D;

  double elapsed, elapsed_memory;

  if (fVerbosity > 0) printf("Running CUDA benchmark...");

  CudaManager::Instance().LoadGeometry(this->GetWorld());
  CudaManager::Instance().Synchronize();
  std::list<CudaVolume> volumesGpu;
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
       v != fVolumes.end(); ++v) {
    volumesGpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->specialized())
      )
    );
  }

  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(fPointCount);

  vecgeom_cuda::Stopwatch timer;
  vecgeom_cuda::Stopwatch timer_memory;

  timer_memory.Start();

  Precision *posXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));

  CudaSOA3D positionGpu  = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);

  Inside_t *insideGpu =
      AllocateOnGpu<Inside_t>(sizeof(Inside_t)*fPointCount);
  
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::InsideBenchmarkCudaKernel<<<launch.grid_size,
                                                launch.block_size>>>(
        *v, positionGpu, fPointCount, insideGpu
      );
    }
  }
  elapsed = timer.Stop();

  CopyFromGpu(insideGpu, inside, fPointCount*sizeof(Inside_t));

  FreeFromGpu(insideGpu);
  FreeFromGpu(posXGpu);
  FreeFromGpu(posYGpu);
  FreeFromGpu(posZGpu);

  elapsed_memory = timer_memory.Stop();

  if (fVerbosity > 0) {
    printf(" Finished in %fs (%fs per volume).\n",
           elapsed, elapsed/fVolumes.size());
  }
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsed, kBenchmarkInside, kBenchmarkCuda, fInsideBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsed_memory, kBenchmarkInside, kBenchmarkCudaMemory, fInsideBias
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

  double elapsedDistance = 0, elapsedDistanceMemory = 0;
  double elapsedSafety = 0, elapsedSafetyMemory = 0;

  if (fVerbosity > 0) printf("Running CUDA benchmark...");

  CudaManager::Instance().LoadGeometry(this->GetWorld());
  CudaManager::Instance().Synchronize();
  std::list<CudaVolume> volumesGpu;
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
       v != fVolumes.end(); ++v) {
    volumesGpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->specialized())
      )
    );
  }

  vecgeom_cuda::LaunchParameters launch =
      vecgeom_cuda::LaunchParameters(fPointCount);
  vecgeom_cuda::Stopwatch timer, timer_memory;

  timer_memory.Start();

  Precision *posXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));
  CudaSOA3D positionGpu  = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);

  timer_memory.Stop();
  elapsedDistanceMemory += timer_memory.Elapsed();
  elapsedSafetyMemory += timer_memory.Elapsed();

  timer_memory.Start();

  Precision *dirXGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirYGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirZGpu = AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  CopyToGpu(dirX, dirXGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirY, dirYGpu, fPointCount*sizeof(Precision));
  CopyToGpu(dirZ, dirZGpu, fPointCount*sizeof(Precision));

  CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, fPointCount);

  elapsedDistanceMemory += timer_memory.Stop();

  Precision *distancesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                     *fPointCount);
  Precision *safetiesGpu = AllocateOnGpu<Precision>(sizeof(Precision)
                                                    *fPointCount);
  
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::DistanceToInBenchmarkCudaKernel<<<launch.grid_size,
                                                      launch.block_size>>>(
        *v, positionGpu, directionGpu, fPointCount, distancesGpu
      );
    }
  }
  elapsedDistance = timer.Stop();

  timer_memory.Start();

  CopyFromGpu(distancesGpu, distances, fPointCount*sizeof(Precision));
  FreeFromGpu(distancesGpu);
  FreeFromGpu(dirXGpu);
  FreeFromGpu(dirYGpu);
  FreeFromGpu(dirZGpu);

  elapsedDistanceMemory += timer_memory.Stop();

  elapsedDistanceMemory += elapsedDistance;
  
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::SafetyToInBenchmarkCudaKernel<<<launch.grid_size,
                                                    launch.block_size>>>(
        *v, positionGpu, fPointCount, safetiesGpu
      );
    }
  }
  elapsedSafety = timer.Stop();

  timer_memory.Start();

  CopyFromGpu(safetiesGpu, safeties, fPointCount*sizeof(Precision));
  FreeFromGpu(safetiesGpu);
  FreeFromGpu(posXGpu);
  FreeFromGpu(posYGpu);
  FreeFromGpu(posZGpu);

  elapsedSafetyMemory += timer_memory.Stop();

  elapsedSafetyMemory += elapsedSafety;

  if (fVerbosity > 0) {
    printf(" Finished in %fs/%fs (%fs/%fs per volume).\n", elapsedDistance,
           elapsedSafety, elapsedDistance/fVolumes.size(),
           elapsedSafety/fVolumes.size());
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
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedDistanceMemory, kBenchmarkDistanceToIn, kBenchmarkCudaMemory,
      fToInBias
    )
  );
  fResults.push_back(
    GenerateBenchmarkResult(
      elapsedSafetyMemory, kBenchmarkSafetyToIn, kBenchmarkCudaMemory,
      fToInBias
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

  if (fVerbosity > 0) printf("Running CUDA benchmark...");

  CudaManager::Instance().LoadGeometry(this->GetWorld());
  CudaManager::Instance().Synchronize();
  std::list<CudaVolume> volumesGpu;
  for (std::list<VolumePointers>::const_iterator v = fVolumes.begin();
       v != fVolumes.end(); ++v) {
    volumesGpu.push_back(
      reinterpret_cast<CudaVolume>(
        CudaManager::Instance().LookupPlaced(v->specialized())
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
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::DistanceToOutBenchmarkCudaKernel<<<launch.grid_size,
                                                       launch.block_size>>>(
        *v, positionGpu, directionGpu, fPointCount, distancesGpu
      );
    }
  }
  elapsedDistance = timer.Stop();
  
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin();
         v != volumesGpu.end(); ++v) {
      vecgeom_cuda::SafetyToOutBenchmarkCudaKernel<<<launch.grid_size,
                                                     launch.block_size>>>(
        *v, positionGpu, fPointCount, safetiesGpu
      );
    }
  }
  elapsedSafety = timer.Stop();

  if (fVerbosity > 0) {
    printf(" Finished in %fs/%fs (%fs/%fs per volume).\n", elapsedDistance,
           elapsedSafety, elapsedDistance/fVolumes.size(),
           elapsedSafety/fVolumes.size());
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
  FreeFromGpu(dirXGpu); FreeFromGpu(dirYGpu);
  FreeFromGpu(dirZGpu);
  
  cudaDeviceSynchronize();
}

} // End namespace vecgeom
