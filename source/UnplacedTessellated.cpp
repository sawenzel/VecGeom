/// @file UnplacedTessellated.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/UnplacedTessellated.h"
#include "VecGeom/volumes/SpecializedTessellated.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#include "VecGeom/base/RNG.h"

#include "VecGeom/management/VolumeFactory.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

void UnplacedTessellated::Print() const
{
  printf("UnplacedTessellated {%zu facets}", fTessellated.fFacets.size());
}

void UnplacedTessellated::Print(std::ostream &os) const
{
  os << "UnplacedTessellated {" << fTessellated.fFacets.size() << " facets " << std::endl;
}

Precision UnplacedTessellated::Capacity() const
{
  if (fTessellated.fCubicVolume != 0.) return fTessellated.fCubicVolume;

  // For explanation of the following algorithm see:
  // https://en.wikipedia.org/wiki/Polyhedron#Volume
  // http://wwwf.imperial.ac.uk/~rn/centroid.pdf

  int size = fTessellated.fFacets.size();
  for (int i = 0; i < size; ++i) {
    TriangleFacet<Precision> &facet = *fTessellated.fFacets[i];
    Precision area                  = facet.fSurfaceArea;
    fTessellated.fCubicVolume += area * (facet.fVertices[0].Dot(facet.fNormal));
  }
  fTessellated.fCubicVolume /= 3.;
  return fTessellated.fCubicVolume;
}

Precision UnplacedTessellated::SurfaceArea() const
{
  if (fTessellated.fSurfaceArea != 0.) return fTessellated.fSurfaceArea;

  int size = fTessellated.fFacets.size();
  for (int i = 0; i < size; ++i) {
    TriangleFacet<Precision> *facet = fTessellated.fFacets[i];
    fTessellated.fSurfaceArea += facet->fSurfaceArea;
  }
  return fTessellated.fSurfaceArea;
}

int UnplacedTessellated::ChooseSurface() const
{
  int choice       = 0; // 0 = zm, 1 = zp, 2 = ym, 3 = yp, 4 = xm, 5 = xp
  Precision Stotal = SurfaceArea();

  // random value to choose surface to place the point
  Precision rand = RNG::Instance().uniform() * Stotal;

  while (rand > fTessellated.fFacets[choice]->fSurfaceArea)
    rand -= fTessellated.fFacets[choice]->fSurfaceArea, choice++;

  return choice;
}

Vector3D<Precision> UnplacedTessellated::SamplePointOnSurface() const
{
  int surface  = ChooseSurface();
  Precision r1 = RNG::Instance().uniform(0.0, 1.0);
  Precision r2 = RNG::Instance().uniform(0.0, 1.0);
  if (r1 + r2 > 1.) {
    r1 = 1. - r1;
    r2 = 1. - r2;
  }

  auto facet = fTessellated.fFacets[surface];
  return (facet->fVertices[0] + r1 * (facet->fVertices[1] - facet->fVertices[0]) +
          r2 * (facet->fVertices[2] - facet->fVertices[0]));
}

bool UnplacedTessellated::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  // Redirect to normal implementation
  bool valid = false;
  norm       = TessellatedImplementation::NormalKernel<Precision>(fTessellated, point, valid);
  return valid;
}

#ifdef VECCORE_CUDA
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedTessellated::Create(LogicalVolume const *const logical_volume,
                                                              Transformation3D const *const transformation,
                                                              const int id, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTessellated(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedTessellated(logical_volume, transformation, id);
}
#else
VPlacedVolume *UnplacedTessellated::Create(LogicalVolume const *const logical_volume,
                                           Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTessellated(logical_volume, transformation);
    return placement;
  }
  return new SpecializedTessellated(logical_volume, transformation);
}
#endif

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedTessellated::SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                      const int id,
#endif
                                                      VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedTessellated>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                    id,
#endif
                                                                    placement);
}

std::ostream &UnplacedTessellated::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Trd\n"
     << " Parameters: \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedTessellated::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
#ifdef HYBRID_NAVIGATOR_PORTED_TO_CUDA
  return CopyToGpuImpl<UnplacedTessellated>(in_gpu_ptr);
#else
  VECGEOM_VALIDATE(0, << "Attempted to copy UnplacedTessellated to GPU.  This is not yet supported.");
  return DevicePtr<cuda::VUnplacedVolume>(nullptr);
#endif
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTessellated::CopyToGpu() const
{
#ifdef HYBRID_NAVIGATOR_PORTED_TO_CUDA
  return CopyToGpuImpl<UnplacedTessellated>();
#else
  VECGEOM_VALIDATE(0, << "Attempted to copy UnplacedTessellated to GPU.  This is not yet supported.");
  return DevicePtr<cuda::VUnplacedVolume>(nullptr);
#endif
}

#ifndef HYBRID_NAVIGATOR_PORTED_TO_CUDA
template <>
size_t DevicePtr<vecgeom::cuda::SpecializedVolImplHelper<vecgeom::cuda::TessellatedImplementation>>::SizeOf()
{
  return 0;
}

template <>
template <>
void DevicePtr<cuda::SpecializedVolImplHelper<cuda::TessellatedImplementation>>::Construct(
    DevicePtr<vecgeom::cuda::LogicalVolume>, DevicePtr<vecgeom::cuda::Transformation3D>, unsigned int, int, int) const
{
  return;
}

template <>
void ConstructManyOnGpu<cuda::SpecializedVolImplHelper<cuda::TessellatedImplementation>
                        /*, ... inferred from arguments */>(
    std::size_t nElement, DevicePtr<cuda::VPlacedVolume> const *gpu_ptrs, DevicePtr<cuda::LogicalVolume> const *logical,
    DevicePtr<cuda::Transformation3D> const *trafo, decltype(std::declval<VPlacedVolume>().id()) const *ids,
    decltype(std::declval<VPlacedVolume>().GetCopyNo()) const *copyNos,
    decltype(std::declval<VPlacedVolume>().GetChildId()) const *childIds)
{
}

#endif

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTessellated>::SizeOf();
template void DevicePtr<cuda::UnplacedTessellated>::Construct() const;
template void ConstructManyOnGpu<cuda::UnplacedTessellated /*, ... inferred from arguments */>(
    std::size_t nElement, DevicePtr<cuda::VPlacedVolume> const *gpu_ptrs, DevicePtr<cuda::LogicalVolume> const *logical,
    DevicePtr<cuda::Transformation3D> const *trafo, decltype(std::declval<VPlacedVolume>().id()) const *ids,
    decltype(std::declval<VPlacedVolume>().GetCopyNo()) const *copyNos,
    decltype(std::declval<VPlacedVolume>().GetChildId()) const *childIds);

} // namespace cxx

#endif

} // namespace vecgeom
