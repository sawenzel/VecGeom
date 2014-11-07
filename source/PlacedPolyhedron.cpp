/// \file PlacedPolyhedron.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#include "volumes/PlacedPolyhedron.h"

#include "volumes/kernel/GenericKernels.h"
#include "volumes/SpecializedPolyhedron.h"

#ifdef VECGEOM_ROOT
#include "TGeoPgon.h"
#endif

#ifdef VECGEOM_USOLIDS
#include "UPolyhedra.hh"
#endif

#ifdef VECGEOM_GEANT4
#include "G4Polyhedra.hh"
#endif

namespace VECGEOM_NAMESPACE {

VECGEOM_CUDA_HEADER_BOTH
int PlacedPolyhedron::PhiSegmentIndex(Vector3D<Precision> const &point) const {
  Vector3D<Precision> localPoint =
     VPlacedVolume::transformation()->Transform(point);
  return PolyhedronImplementation<true>::ScalarFindPhiSegment(
      *GetUnplacedVolume(), localPoint);
}

#ifndef VECGEOM_NVCC

VPlacedVolume const* PlacedPolyhedron::ConvertToUnspecialized() const {
  return new SimplePolyhedron(GetLabel().c_str(), logical_volume(),
                              transformation());
}

#ifdef VECGEOM_ROOT
TGeoShape const* PlacedPolyhedron::ConvertToRoot() const {

  const int zPlaneCount = GetZSegmentCount()+1;

  TGeoPgon *pgon = new TGeoPgon(
      GetLabel().c_str(), GetPhiStart(), GetPhiDelta(), GetSideCount(),
      zPlaneCount);

  // Define sections of TGeoPgon. It takes care of the rest internally once the
  // last section is set.
  for (int i = 0; i < zPlaneCount; ++i) {
    pgon->DefineSection(i, GetZPlanes()[i], GetRMin()[i], GetRMax()[i]);
  }

  return pgon;
}
#endif

#ifdef VECGEOM_USOLIDS
::VUSolid const* PlacedPolyhedron::ConvertToUSolids() const {

  return new UPolyhedra(
      GetLabel().c_str(),
      kDegToRad*GetPhiStart(),
      kDegToRad*GetPhiDelta(),
      GetSideCount(),
      GetZSegmentCount()+1,
      &GetZPlanes()[0],
      &GetRMin()[0],
      &GetRMax()[0]);
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const* PlacedPolyhedron::ConvertToGeant4() const {

  return new G4Polyhedra(
      GetLabel().c_str(),
      kDegToRad*GetPhiStart(),
      kDegToRad*GetPhiDelta(),
      GetSideCount(),
      GetZSegmentCount()+1,
      &GetZPlanes()[0],
      &GetRMin()[0],
      &GetRMax()[0]);
}
#endif

#endif // !VECGEOM_NVCC

} // End global namespace

namespace vecgeom {

#ifdef VECGEOM_CUDA_INTERFACE

void PlacedPolyhedron_CopyToGpu(LogicalVolume const *const logical_volume,
                                Transformation3D const *const transformation,
                                const int id,
                                VPlacedVolume *const gpu_ptr);

VPlacedVolume* PlacedPolyhedron::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation,
    VPlacedVolume *const gpu_ptr) const {
  vecgeom::PlacedPolyhedron_CopyToGpu(logical_volume, transformation,
                                      VPlacedVolume::id(), gpu_ptr);
  vecgeom::CudaAssertError();
  return gpu_ptr;
}

VPlacedVolume* PlacedPolyhedron::CopyToGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation) const {
  VPlacedVolume *const gpu_ptr = vecgeom::AllocateOnGpu<PlacedPolyhedron>();
  return this->CopyToGpu(logical_volume, transformation, gpu_ptr);
}

#endif // VECGEOM_CUDA_INTERFACE

#ifdef VECGEOM_NVCC

class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

__global__
void PlacedPolyhedron_ConstructOnGpu(
    LogicalVolume const *const logical_volume,
    Transformation3D const *const transformation, const int id,
    VPlacedVolume *const gpu_ptr) {
  new(gpu_ptr) vecgeom_cuda::SimplePolyhedron(
    reinterpret_cast<vecgeom_cuda::LogicalVolume const*>(logical_volume),
    reinterpret_cast<vecgeom_cuda::Transformation3D const*>(transformation),
    id
  );
}

void PlacedPolyhedron_CopyToGpu(LogicalVolume const *const logical_volume,
                                Transformation3D const *const transformation,
                                const int id, VPlacedVolume *const gpu_ptr) {
  PlacedPolyhedron_ConstructOnGpu<<<1, 1>>>(logical_volume, transformation, id,
                                            gpu_ptr);
}

#endif // VECGEOM_NVCC

} // End namespace vecgeom
