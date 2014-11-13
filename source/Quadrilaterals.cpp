#include "volumes/Quadrilaterals.h"

#include "backend/scalar/Backend.h"

#include <memory>

namespace VECGEOM_NAMESPACE {

#ifdef VECGEOM_STD_CXX11
Quadrilaterals::Quadrilaterals(int size)
    : fPlanes(size), fSideVectors{size, size, size, size},
      fCorners{size, size, size, size} {}
#endif

#ifdef VECGEOM_NVCC
__device__
Quadrilaterals::Quadrilaterals(
    Precision *planeA, Precision *planeB, Precision *planeC,
    Precision *planeD, Precision *side0A, Precision *side0B,
    Precision *side0C, Precision *side0D, Precision *side1A,
    Precision *side1B, Precision *side1C, Precision *side1D,
    Precision *side2A, Precision *side2B, Precision *side2C,
    Precision *side2D, Precision *side3A, Precision *side3B,
    Precision *side3C, Precision *side3D,
    AOS3D<Precision> const &corner0, AOS3D<Precision> const &corner1,
    AOS3D<Precision> const &corner2, AOS3D<Precision> const &corner3, int size)
    : fPlanes(planeA, planeB, planeC, planeD, size) {
  fSideVectors[0] = Planes(side0A, side0B, side0C, side0D, size);
  fSideVectors[1] = Planes(side1A, side1B, side1C, side1D, size);
  fSideVectors[2] = Planes(side2A, side2B, side2C, side2D, size);
  fSideVectors[3] = Planes(side3A, side3B, side3C, side3D, size);
  fCorners[0] = corner0;
  fCorners[1] = corner1;
  fCorners[2] = corner2;
  fCorners[3] = corner3;
}
#endif

VECGEOM_CUDA_HEADER_BOTH
Quadrilaterals::~Quadrilaterals() {}

Quadrilaterals& Quadrilaterals::operator=(Quadrilaterals const &other) {
  fPlanes = other.fPlanes;
  for (int i = 0; i < 4; ++i) {
    fSideVectors[i] = other.fSideVectors[i];
    fCorners[i] = other.fCorners[i];
  }
  return *this;
}

#ifdef VECGEOM_STD_CXX11
void Quadrilaterals::Set(
    int index,
    Vector3D<Precision> const &corner0,
    Vector3D<Precision> const &corner1,
    Vector3D<Precision> const &corner2,
    Vector3D<Precision> const &corner3) {

  // TODO: It should be asserted that the quadrilateral is planar and convex.

  fCorners[0].set(index, corner0);
  fCorners[1].set(index, corner1);
  fCorners[2].set(index, corner2);
  fCorners[3].set(index, corner3);

  // Compute plane equation to retrieve normal and distance to origin
  // ax + by + cz + d = 0
  Precision a, b, c, d;
  a = corner0[1]*(corner1[2] - corner2[2]) +
      corner1[1]*(corner2[2] - corner0[2]) +
      corner2[1]*(corner0[2] - corner1[2]);
  b = corner0[2]*(corner1[0] - corner2[0]) +
      corner1[2]*(corner2[0] - corner0[0]) +
      corner2[2]*(corner0[0] - corner1[0]);
  c = corner0[0]*(corner1[1] - corner2[1]) +
      corner1[0]*(corner2[1] - corner0[1]) +
      corner2[0]*(corner0[1] - corner1[1]);
  d = - corner0[0]*(corner1[1]*corner2[2] - corner2[1]*corner1[2])
      - corner1[0]*(corner2[1]*corner0[2] - corner0[1]*corner2[2])
      - corner2[0]*(corner0[1]*corner1[2] - corner1[1]*corner0[2]);
  Vector3D<Precision> normal(a, b, c);
  // Normalize the plane equation
  // (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2) = 0 =>
  // n0*x + n1*x + n2*x + p = 0
  Precision inverseLength = 1. / normal.Length();
  normal *= inverseLength;
  d *= inverseLength;

  fPlanes.Set(index, normal, d);

  auto ComputeSideVector = [&index, &normal] (
      Planes &sideVectors,
      Vector3D<Precision> const &c0,
      Vector3D<Precision> const &c1) {
    Vector3D<Precision> sideVector = normal.Cross(c1-c0).Normalized();
    sideVectors.Set(index, sideVector, c0);
  };

  ComputeSideVector(fSideVectors[0], corner0, corner1);
  ComputeSideVector(fSideVectors[1], corner1, corner2);
  ComputeSideVector(fSideVectors[2], corner2, corner3);
  ComputeSideVector(fSideVectors[3], corner3, corner0);
}
#endif

void Quadrilaterals::FlipSign(int index) {
  fPlanes.FlipSign(index);
}

VECGEOM_CUDA_HEADER_BOTH
void Quadrilaterals::Print() const {
  for (int i = 0, iMax = size(); i < iMax; ++i) {
    printf("{(%.2f, %.2f, %.2f, %.2f), {", GetNormals().x(i),
           GetNormals().y(i), GetNormals().z(i), GetDistance(i));
    for (int j = 0; j < 3; ++j) {
      printf("(%.2f, %.2f, %.2f, %.2f), ",
             GetSideVectors()[j].GetNormals().x(i),
             GetSideVectors()[j].GetNormals().y(i),
             GetSideVectors()[j].GetNormals().z(i),
             GetSideVectors()[j].GetDistance(i));
    }
    printf("(%.2f, %.2f, %.2f, %.2f)}}",
           GetSideVectors()[3].GetNormals().x(i),
           GetSideVectors()[3].GetNormals().y(i),
           GetSideVectors()[3].GetNormals().z(i),
           GetSideVectors()[3].GetDistance(i));
  }
}

std::ostream& operator<<(std::ostream &os, Quadrilaterals const &quads) {
  for (int i = 0, iMax = quads.size(); i < iMax; ++i) {
    os << "{(" << quads.GetNormal(i) << ", " << quads.GetDistance(i)
       << "), {(";
    for (int j = 0; j < 3; ++j) {
      os << quads.GetSideVectors()[j].GetNormals()[i]
         << ", " << quads.GetSideVectors()[j].GetDistances()[i] << "), ";
    }
    os << quads.GetSideVectors()[3].GetNormals()[i]
       << ", " << quads.GetSideVectors()[3].GetDistances()[i] << ")}}\n";
  }
  return os;
}

} // End global namespace

#ifdef VECGEOM_CUDA_INTERFACE
namespace vecgeom_cuda {
template <typename T> class Vector3D;
}
#endif

namespace vecgeom {

#ifdef VECGEOM_CUDA

#ifdef VECGEOM_NVCC
template <typename T> class Vector3D;
#endif

void Quadrilaterals_AllocateCornersOnGpu(
    vecgeom_cuda::Vector3D<Precision> *corners[4], int size);

void Quadrilaterals_CopyToGpu(
    void *gpuPtr,
    Precision *planeA, Precision *planeB, Precision *planeC, Precision *planeD,
    Precision *side0A, Precision *side0B, Precision *side0C, Precision *side0D,
    Precision *side1A, Precision *side1B, Precision *side1C, Precision *side1D,
    Precision *side2A, Precision *side2B, Precision *side2C, Precision *side2D,
    Precision *side3A, Precision *side3B, Precision *side3C, Precision *side3D,
    Precision *corner0Array, Precision *corner1Array,
    Precision *corner2Array, Precision *corner3Array,
    vecgeom_cuda::Vector3D<Precision>* corner0,
    vecgeom_cuda::Vector3D<Precision>* corner1,
    vecgeom_cuda::Vector3D<Precision>* corner2,
    vecgeom_cuda::Vector3D<Precision>* corner3, int size);

#ifdef VECGEOM_CUDA_INTERFACE

  void Quadrilaterals::CopyToGpu(void *gpuPtr) const {
    size_t bytes = size()*sizeof(Precision);
    size_t vecBytes = 3*size()*sizeof(Precision);
    Precision *plane[4];
    Precision *sides[4][4];
    Precision *cornerData[4];
    vecgeom_cuda::Vector3D<Precision> *corners[4];
    // TODO: no one has responsibility for cleaning this up!! This has to be
    //       delegated rather urgently.
    Quadrilaterals_AllocateCornersOnGpu(corners, size());
    for (int i = 0; i < 4; ++i) {
      plane[i] = AllocateOnGpu<Precision>(bytes);
      vecgeom::CopyToGpu(fPlanes[i], plane[i], bytes);
      for (int j = 0; j < 4; ++j) {
        sides[i][j] = AllocateOnGpu<Precision>(bytes);
        vecgeom::CopyToGpu(fSideVectors[i][j], sides[i][j], bytes);
      }
      // Form a contiguous C-array of corner coordinates to construct the AOS
      // on the GPU
      cornerData[i] = AllocateOnGpu<Precision>(vecBytes);
      std::unique_ptr<Precision[]> cornerArray(new Precision[3*size()]);
      for (int j = 0; j < size(); ++j) {
        cornerArray[3*j]   = fCorners[i].x(j);
        cornerArray[3*j+1] = fCorners[i].y(j);
        cornerArray[3*j+2] = fCorners[i].z(j);
      }
      vecgeom::CopyToGpu(cornerArray.get(), cornerData[i], vecBytes);
    }
    Quadrilaterals_CopyToGpu(
        gpuPtr, plane[0], plane[1], plane[2], plane[3],
        sides[0][0], sides[0][1], sides[0][2], sides[0][3],
        sides[1][0], sides[1][1], sides[1][2], sides[1][3],
        sides[2][0], sides[2][1], sides[2][2], sides[2][3],
        sides[3][0], sides[3][1], sides[3][2], sides[3][3],
        cornerData[0], cornerData[1], cornerData[2], cornerData[3],
        corners[0], corners[1], corners[2], corners[3],
        size());
  }

#endif // end VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

void Quadrilaterals_AllocateCornersOnGpu(
    vecgeom_cuda::Vector3D<Precision> *corners[4], int size) {
  // TODO: no one has responsibility for cleaning this up!! This has to be
  //       delegated rather urgently
  for (int i = 0; i < 4; ++i) {
    corners[i] = AllocateOnGpu<vecgeom_cuda::Vector3D<Precision> >(
        size*sizeof(vecgeom_cuda::Vector3D<Precision>));
  }
}

__global__
void Quadrilaterals_ConstructOnGpu(
    vecgeom_cuda::Quadrilaterals *gpuPtr,
    Precision *planeA, Precision *planeB, Precision *planeC, Precision *planeD,
    Precision *side0A, Precision *side0B, Precision *side0C, Precision *side0D,
    Precision *side1A, Precision *side1B, Precision *side1C, Precision *side1D,
    Precision *side2A, Precision *side2B, Precision *side2C, Precision *side2D,
    Precision *side3A, Precision *side3B, Precision *side3C, Precision *side3D,
    Precision *corner0Array, Precision *corner1Array,
    Precision *corner2Array, Precision *corner3Array,
    vecgeom_cuda::Vector3D<Precision> *corner0,
    vecgeom_cuda::Vector3D<Precision> *corner1,
    vecgeom_cuda::Vector3D<Precision> *corner2,
    vecgeom_cuda::Vector3D<Precision> *corner3,
    const int size) {
  for (int i = 0; i < size; ++i) {
    new (corner0+i) vecgeom_cuda::Vector3D<Precision>(
        corner0Array[3*i], corner0Array[3*i+1], corner0Array[3*i+2]);
    new (corner1+i) vecgeom_cuda::Vector3D<Precision>(
        corner1Array[3*i], corner1Array[3*i+1], corner1Array[3*i+2]);
    new (corner2+i) vecgeom_cuda::Vector3D<Precision>(
        corner2Array[3*i], corner2Array[3*i+1], corner2Array[3*i+2]);
    new (corner3+i) vecgeom_cuda::Vector3D<Precision>(
        corner3Array[3*i], corner3Array[3*i+1], corner3Array[3*i+2]);
  }
  new (gpuPtr) vecgeom_cuda::Quadrilaterals(
      planeA, planeB, planeC, planeD,
      side0A, side0B, side0C, side0D,
      side1A, side1B, side1C, side1D,
      side2A, side2B, side2C, side2D,
      side3A, side3B, side3C, side3D,
      vecgeom_cuda::AOS3D<Precision>(corner0, size),
      vecgeom_cuda::AOS3D<Precision>(corner1, size),
      vecgeom_cuda::AOS3D<Precision>(corner2, size),
      vecgeom_cuda::AOS3D<Precision>(corner3, size), size);
}

void Quadrilaterals_CopyToGpu(
    void *gpuPtr,
    Precision *planeA, Precision *planeB, Precision *planeC, Precision *planeD,
    Precision *side0A, Precision *side0B, Precision *side0C, Precision *side0D,
    Precision *side1A, Precision *side1B, Precision *side1C, Precision *side1D,
    Precision *side2A, Precision *side2B, Precision *side2C, Precision *side2D,
    Precision *side3A, Precision *side3B, Precision *side3C, Precision *side3D,
    Precision *corner0Array, Precision *corner1Array,
    Precision *corner2Array, Precision *corner3Array,
    vecgeom_cuda::Vector3D<Precision>* corner0,
    vecgeom_cuda::Vector3D<Precision>* corner1,
    vecgeom_cuda::Vector3D<Precision>* corner2,
    vecgeom_cuda::Vector3D<Precision>* corner3, int size) {
  Quadrilaterals_ConstructOnGpu<<<1, 1>>>(
      static_cast<vecgeom_cuda::Quadrilaterals*>(gpuPtr),
      planeA, planeB, planeC, planeD,
      side0A, side0B, side0C, side0D,
      side1A, side1B, side1C, side1D,
      side2A, side2B, side2C, side2D,
      side3A, side3B, side3C, side3D,
      corner0Array, corner1Array, corner2Array, corner3Array,
      corner0, corner1, corner2, corner3, size);
}

#endif // VECGEOM_NVCC

#endif // VECGEOM_CUDA

} // End namespace vecgeom
