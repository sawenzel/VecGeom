/// \file Benchmarker.cu
/// \author Johannes de Fine Licht

#include "benchmarking/Benchmarker.h"

#include "base/Stopwatch.h"
#include "backend/cuda/Backend.h"
#include "management/CudaManager.h"

namespace vecgeom {
inline namespace cuda {

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

} // End cuda namespace

// This is odd ... We are implementing one of the member function of cxx class
// in a .cu file

void Benchmarker::RunInsideCuda(
    Precision *const posX, Precision *const posY,
    Precision *const posZ, bool *const contains, Inside_t *const inside) {

  typedef cxx::DevicePtr<vecgeom::cuda::VPlacedVolume> CudaVolume;
  typedef vecgeom::cuda::SOA3D<Precision> CudaSOA3D;

  if (fVerbosity > 0) printf("CUDA          - ");

  std::list<CudaVolume> volumesGpu;
  GetVolumePointers( volumesGpu);

  vecgeom::cuda::LaunchParameters launch =
      vecgeom::cuda::LaunchParameters(fPointCount);

  Stopwatch timer;

  cxx::DevicePtr<Precision> posXGpu; posXGpu.Allocate(fPointCount);
  cxx::DevicePtr<Precision> posYGpu; posYGpu.Allocate(fPointCount);
  cxx::DevicePtr<Precision> posZGpu; posZGpu.Allocate(fPointCount);
  posXGpu.ToDevice(posX, fPointCount);
  posYGpu.ToDevice(posY, fPointCount);
  posZGpu.ToDevice(posZ, fPointCount);

  bool *containsGpu   = cxx::AllocateOnGpu<bool>(sizeof(bool)*fPointCount);
  Inside_t *insideGpu = cxx::AllocateOnGpu<Inside_t>(sizeof(Inside_t)*fPointCount);

  CudaSOA3D positionGpu  = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom::cuda::InsideBenchmarkCudaKernel<<<launch.grid_size,
                                                launch.block_size>>>(
        *v, positionGpu, fPointCount, insideGpu
      );
    }
  }
  cudaDeviceSynchronize();
  Precision elapsedInside = timer.Stop();

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom::cuda::ContainsBenchmarkCudaKernel<<<launch.grid_size,
                                                  launch.block_size>>>(
        *v, positionGpu, fPointCount, containsGpu
      );
    }
  }
  cudaDeviceSynchronize();
  Precision elapsedContains = timer.Stop();

  cxx::CopyFromGpu(insideGpu, inside, fPointCount*sizeof(Inside_t));
  cxx::CopyFromGpu(containsGpu, contains, fPointCount*sizeof(bool));
  cxx::FreeFromGpu(containsGpu);

  cxx::FreeFromGpu(insideGpu);
  posXGpu.Deallocate();
  posYGpu.Deallocate();
  posZGpu.Deallocate();

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

  typedef cxx::DevicePtr<vecgeom::cuda::VPlacedVolume> CudaVolume;
  typedef vecgeom::cuda::SOA3D<Precision> CudaSOA3D;

  if (fVerbosity > 0) printf("CUDA          - ");

  std::list<CudaVolume> volumesGpu;
  GetVolumePointers( volumesGpu);

  vecgeom::cuda::LaunchParameters launch =
      vecgeom::cuda::LaunchParameters(fPointCount);
  vecgeom::cuda::Stopwatch timer;

  Precision *posXGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  cxx::CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));
  CudaSOA3D positionGpu = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);

  Precision *dirXGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirYGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirZGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  cxx::CopyToGpu(dirX, dirXGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(dirY, dirYGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(dirZ, dirZGpu, fPointCount*sizeof(Precision));
  CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, fPointCount);

  Precision *distancesGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)
                                                     *fPointCount);
  Precision *safetiesGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)
                                                    *fPointCount);

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom::cuda::SafetyToInBenchmarkCudaKernel<<<launch.grid_size,
                                                    launch.block_size>>>(
        *v, positionGpu, fPointCount, safetiesGpu
      );
    }
  }
  cudaDeviceSynchronize();
  Precision elapsedSafety = timer.Stop();

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom::cuda::DistanceToInBenchmarkCudaKernel<<<launch.grid_size,
                                                      launch.block_size>>>(
        *v, positionGpu, directionGpu, fPointCount, distancesGpu
      );
    }
  }
  cudaDeviceSynchronize();
  Precision elapsedDistance = timer.Stop();

  cxx::CopyFromGpu(safetiesGpu, safeties, fPointCount*sizeof(Precision));
  cxx::CopyFromGpu(distancesGpu, distances, fPointCount*sizeof(Precision));
  cxx::FreeFromGpu(distancesGpu);
  cxx::FreeFromGpu(dirXGpu);
  cxx::FreeFromGpu(dirYGpu);
  cxx::FreeFromGpu(dirZGpu);
  cxx::FreeFromGpu(safetiesGpu);
  cxx::FreeFromGpu(posXGpu);
  cxx::FreeFromGpu(posYGpu);
  cxx::FreeFromGpu(posZGpu);


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

  typedef cxx::DevicePtr<vecgeom::cuda::VPlacedVolume> CudaVolume;
  typedef vecgeom::cuda::SOA3D<Precision> CudaSOA3D;

  double elapsedDistance;
  double elapsedSafety;

  if (fVerbosity > 0) printf("CUDA          - ");

  std::list<CudaVolume> volumesGpu;
  GetVolumePointers( volumesGpu);

  Precision *posXGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posYGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *posZGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirXGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirYGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  Precision *dirZGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)*fPointCount);
  cxx::CopyToGpu(posX, posXGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(posY, posYGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(posZ, posZGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(dirX, dirXGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(dirY, dirYGpu, fPointCount*sizeof(Precision));
  cxx::CopyToGpu(dirZ, dirZGpu, fPointCount*sizeof(Precision));

  CudaSOA3D positionGpu  = CudaSOA3D(posXGpu, posYGpu, posZGpu, fPointCount);
  CudaSOA3D directionGpu = CudaSOA3D(dirXGpu, dirYGpu, dirZGpu, fPointCount);

  Precision *distancesGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)
                                                     *fPointCount);
  Precision *safetiesGpu = cxx::AllocateOnGpu<Precision>(sizeof(Precision)
                                                    *fPointCount);

  vecgeom::cuda::LaunchParameters launch =
      vecgeom::cuda::LaunchParameters(fPointCount);
  vecgeom::cuda::Stopwatch timer;
  
  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom::cuda::SafetyToOutBenchmarkCudaKernel<<<launch.grid_size,
                                                     launch.block_size>>>(
        *v, positionGpu, fPointCount, safetiesGpu
      );
    }
  }
  cudaDeviceSynchronize();
  elapsedSafety = timer.Stop();

  timer.Start();
  for (unsigned r = 0; r < fRepetitions; ++r) {
    for (std::list<CudaVolume>::const_iterator v = volumesGpu.begin(),
         vEnd = volumesGpu.end(); v != vEnd; ++v) {
      vecgeom::cuda::DistanceToOutBenchmarkCudaKernel<<<launch.grid_size,
                                                       launch.block_size>>>(
        *v, positionGpu, directionGpu, fPointCount, distancesGpu
      );
    }
  }
  cudaDeviceSynchronize();
  elapsedDistance = timer.Stop();

  cxx::CopyFromGpu(distancesGpu, distances, fPointCount*sizeof(Precision));
  cxx::CopyFromGpu(safetiesGpu, safeties, fPointCount*sizeof(Precision));

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

  cxx::FreeFromGpu(safetiesGpu);
  cxx::FreeFromGpu(posXGpu);
  cxx::FreeFromGpu(posYGpu);
  cxx::FreeFromGpu(posZGpu);
  cxx::FreeFromGpu(dirXGpu);
  cxx::FreeFromGpu(dirYGpu);
  cxx::FreeFromGpu(dirZGpu);
  
  cudaDeviceSynchronize();
}

} // End vecgeom namespace

