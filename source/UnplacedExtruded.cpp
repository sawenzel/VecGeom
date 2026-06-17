/// @file UnplacedExtruded.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/Tessellated.h"
#include "VecGeom/volumes/UnplacedExtruded.h"
#include "VecGeom/volumes/UnplacedSExtruVolume.h"
#include "VecGeom/volumes/SpecializedExtruded.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#include "VecGeom/base/RNG.h"

#include "VecGeom/management/VolumeFactory.h"
#include <cstddef> // offsetof

#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedImplAs.h"
#endif

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
#include "TGeoXtru.h"
#endif

#ifdef VECGEOM_GEANT4
#include "G4ExtrudedSolid.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#endif
#endif

namespace vecgeom {

#ifdef VECCORE_CUDA
inline
#endif
    namespace cuda {

template <typename Real_t>
BVH<Real_t> *AllocateDeviceBVHBuffer(size_t n);
} // namespace cuda

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
TGeoShape const *UnplacedExtruded::ConvertToRoot(char const *label) const
{
  size_t nvert = GetNVertices();
  size_t nsect = GetNSections();

  // if(nsect > 1){
  double *x = new double[nvert];
  double *y = new double[nvert];
  for (size_t i = 0; i < nvert; ++i) {
    Precision xcrt, ycrt;
    GetVertex(i, xcrt, ycrt);
    x[i] = xcrt;
    y[i] = ycrt;
  }
  TGeoXtru *xtru = new TGeoXtru(nsect);
  xtru->DefinePolygon(nvert, x, y);
  for (size_t i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    xtru->DefineSection(i, sect.fOrigin.z(), sect.fOrigin.x(), sect.fOrigin.y(), sect.fScale);
  }
  delete[] x;
  delete[] y;
  return xtru;
}
#endif

#ifdef VECGEOM_GEANT4
G4VSolid const *UnplacedExtruded::ConvertToGeant4(char const *label) const
{
  std::vector<G4TwoVector> polygon;
  Precision x, y;
  size_t nvert = GetNVertices();
  for (size_t i = 0; i < nvert; ++i) {
    GetVertex(i, x, y);
    polygon.push_back(G4TwoVector(x, y));
  }
  std::vector<G4ExtrudedSolid::ZSection> sections;
  size_t nsect = GetNSections();
  for (size_t i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    sections.push_back(
        G4ExtrudedSolid::ZSection(sect.fOrigin.z(), G4TwoVector(sect.fOrigin.x(), sect.fOrigin.y()), sect.fScale));
  }
  G4ExtrudedSolid *g4xtru = new G4ExtrudedSolid(label, polygon, sections);
  return g4xtru;
}
#endif
#endif

template <>
UnplacedExtruded *Maker<UnplacedExtruded>::MakeInstance(const size_t nvertices, XtruVertex2 const *vertices,
                                                        const int nsections, XtruSection const *sections)
{

#ifndef VECGEOM_NO_SPECIALIZATION
  bool isSExtru = false;
  for (int i = 0; i < (nsections - 1); i++) {
    if (i == 0) {
      isSExtru = ((sections[i].fOrigin - sections[i + 1].fOrigin).Perp2() < kTolerance &&
                  vecCore::math::Abs(sections[i].fScale - sections[i + 1].fScale) < kTolerance);
    } else {
      isSExtru &= ((sections[i].fOrigin - sections[i + 1].fOrigin).Perp2() < kTolerance &&
                   vecCore::math::Abs(sections[i].fScale - sections[i + 1].fScale) < kTolerance);
    }
    if (!isSExtru) break;
  }
  if (isSExtru) {
    Precision *x = new Precision[nvertices];
    Precision *y = new Precision[nvertices];
    for (size_t i = 0; i < nvertices; ++i) {
      x[i] = vertices[i].x;
      y[i] = vertices[i].y;
    }
    Precision zmin = sections[0].fOrigin.z();
    Precision zmax = sections[nsections - 1].fOrigin.z();
    return new SUnplacedImplAs<UnplacedExtruded, UnplacedSExtruVolume>(nvertices, x, y, zmin, zmax);
  } else {
    return new UnplacedExtruded(nvertices, vertices, nsections, sections);
  }
#else
  return new UnplacedExtruded(nvertices, vertices, nsections, sections);
#endif
}

void UnplacedExtruded::Print() const
{
#ifdef VECCORE_CUDA
  printf("UnplacedExtruded");
#else
  std::cerr << "UnplacedExtruded: vertices {";
  int nvert = GetNVertices();
  Precision x, y;
  for (int i = 0; i < nvert - 1; ++i) {
    GetVertex(i, x, y);
    std::cerr << "(" << x << ", " << y << "), ";
  }
  GetVertex(nvert - 1, x, y);
  std::cerr << "(" << x << ", " << y << ")}\n";
  std::cerr << "sections:\n";
  int nsect = GetNSections();
  for (int i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    std::cerr << "orig: (" << sect.fOrigin.x() << ", " << sect.fOrigin.y() << ", " << sect.fOrigin.z()
              << ") scl = " << sect.fScale << std::endl;
  }
#endif
}

void UnplacedExtruded::Print(std::ostream &os) const
{
  os << "UnplacedExtruded: vertices {";
  int nvert = GetNVertices();
  Precision x, y;
  for (int i = 0; i < nvert - 1; ++i) {
    GetVertex(i, x, y);
    os << "(" << x << ", " << y << "), ";
  }
  GetVertex(nvert - 1, x, y);
  os << "(" << x << ", " << y << ")}\n";
  os << "sections:\n";
  int nsect = GetNSections();
  for (int i = 0; i < nsect; ++i) {
    XtruSection sect = GetSection(i);
    os << "orig: (" << sect.fOrigin.x() << ", " << sect.fOrigin.y() << ", " << sect.fOrigin.z()
       << ") scl = " << sect.fScale << std::endl;
  }
}

void UnplacedExtruded::Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const
{
  // TODO: this differentiated treatment seems not to be needed
  if (fXtru.fIsSxtru) {
    fXtru.fSxtruHelper.Extent(aMin, aMax);
  } else {
    aMin = fXtru.fTslRuntimeHelper.fMinExtent;
    aMax = fXtru.fTslRuntimeHelper.fMaxExtent;
  }
}

Precision UnplacedExtruded::Capacity() const
{
  if (fXtru.fCubicVolume != 0.) return fXtru.fCubicVolume;

  if (fXtru.fIsSxtru) {
    fXtru.fCubicVolume =
        fXtru.fSxtruHelper.GetPolygon().Area() * (fXtru.fSxtruHelper.GetUpperZ() - fXtru.fSxtruHelper.GetLowerZ());
  } else {
    int size = fXtru.fTslRuntimeHelper.fNFacets;
    for (int i = 0; i < size; ++i) {
      const auto &facet = fXtru.fTslRuntimeHelper.fFacets[i];
      Precision area    = facet.SurfaceArea();
      fXtru.fCubicVolume += area * (facet.fVertices[0].Dot(facet.fNormal));
    }
    fXtru.fCubicVolume /= 3.;
  }
  return fXtru.fCubicVolume;
}

Precision UnplacedExtruded::SurfaceArea() const
{
  if (fXtru.fSurfaceArea != 0.) return fXtru.fSurfaceArea;

  if (fXtru.fIsSxtru) {
    fXtru.fSurfaceArea = fXtru.fSxtruHelper.SurfaceArea() + 2. * fXtru.fSxtruHelper.GetPolygon().Area();
  } else {
    int size = fXtru.fTslRuntimeHelper.fNFacets;
    for (int i = 0; i < size; ++i) {
      const auto &facet = fXtru.fTslRuntimeHelper.fFacets[i];
      fXtru.fSurfaceArea += facet.SurfaceArea();
    }
  }
  return fXtru.fSurfaceArea;
}

int UnplacedExtruded::ChooseSurface() const
{
  int choice       = 0; // 0 = zm, 1 = zp, 2 = ym, 3 = yp, 4 = xm, 5 = xp
  Precision Stotal = SurfaceArea();

  // random value to choose surface to place the point
  Precision rand = RNG::Instance().uniform() * Stotal;

  while (rand > fXtru.fTslRuntimeHelper.fFacets[choice].SurfaceArea())
    rand -= fXtru.fTslRuntimeHelper.fFacets[choice].SurfaceArea(), choice++;

  return choice;
}

Vector3D<Precision> UnplacedExtruded::SamplePointOnSurface() const
{
  int surface  = ChooseSurface();
  Precision r1 = RNG::Instance().uniform(0.0, 1.0);
  Precision r2 = RNG::Instance().uniform(0.0, 1.0);
  if (r1 + r2 > 1.) {
    r1 = 1. - r1;
    r2 = 1. - r2;
  }
  const auto &facet = fXtru.fTslRuntimeHelper.fFacets[surface];
  return (facet.fVertices[0] + r1 * (facet.fVertices[1] - facet.fVertices[0]) +
          r2 * (facet.fVertices[2] - facet.fVertices[0]));
}

bool UnplacedExtruded::Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const
{
  // Redirect to normal implementation
  bool valid = false;
  if (fXtru.fIsSxtru) {
    norm = SExtruImplementation::NormalKernel(fXtru.fSxtruHelper, point, valid);
  } else {
    norm = TessellatedImplementation::NormalKernel<Precision>(fXtru.fTslRuntimeHelper, point, valid);
  }
  return valid;
}

#ifndef VECCORE_CUDA
SolidMesh *UnplacedExtruded::CreateMesh3D(Transformation3D const &trans, size_t nSegments) const
{

  typedef Vector3D<Precision> Vec_t;

  SolidMesh *sm = new SolidMesh();

  size_t n         = GetNVertices();
  size_t nSections = GetNSections();

  Vec_t *vertices = new Vec_t[nSections * (n + 1)];
  size_t idx      = 0;
  for (size_t i = 0; i < nSections; i++) {
    for (size_t j = n; j > 0; j--) {
      vertices[idx++] = GetStruct().VertexToSection(j - 1, i);
    }
    vertices[idx++] = GetStruct().VertexToSection(n - 1, i);
  }

  sm->ResetMesh(nSections * (n + 1), (nSections - 1) * (n) + 2);
  sm->SetVertices(vertices, nSections * (n + 1));
  delete[] vertices;
  sm->TransformVertices(trans);

  std::vector<size_t> indices;
  for (size_t i = n; i > 0; i--) {
    indices.push_back(i - 1);
  }
  sm->AddPolygon(n, indices, GetStruct().IsConvexPolygon());

  indices.clear();

  for (size_t i = 0, k = (nSections - 1) * (n + 1); i < n; i++, k++) {
    indices.push_back(k);
  }

  sm->AddPolygon(n, indices, GetStruct().IsConvexPolygon());

  size_t k = 0;
  for (size_t i = 0; i < nSections - 1; i++, k++) {
    for (size_t j = 0; j < n; j++, k++) {
      sm->AddPolygon(4, {k, k + 1, k + 1 + n + 1, k + n + 1}, true);
    }
  }

  return sm;
}
#endif

#ifdef VECCORE_CUDA
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedExtruded::Create(LogicalVolume const *const logical_volume,
                                                           Transformation3D const *const transformation, const int id,
                                                           const int copy_no, const int child_id,
                                                           VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedExtruded(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedExtruded(logical_volume, transformation, id, copy_no, child_id);
}
#else
VPlacedVolume *UnplacedExtruded::Create(LogicalVolume const *const logical_volume,
                                        Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedExtruded(logical_volume, transformation);
    return placement;
  }
  return new SpecializedExtruded(logical_volume, transformation);
}
#endif

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedExtruded::SpecializedVolume(LogicalVolume const *const volume,
                                                   Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                   const int id, const int copy_no, const int child_id,
#endif
                                                   VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedExtruded>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                 id, copy_no, child_id,
#endif
                                                                 placement);
}

std::ostream &UnplacedExtruded::StreamInfo(std::ostream &os) const
{
  int oldprc = os.precision(16);
  os << "-----------------------------------------------------------\n"
     << "     *** Dump for solid - " << GetEntityType() << " ***\n"
     << "     ===================================================\n"
     << " Solid type: Extruded\n"
     << " Parameters: \n"
     << "-----------------------------------------------------------\n";
  os.precision(oldprc);
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedExtruded::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  if (GetStruct().fIsSxtru) {
    auto const &shell      = GetStruct().fSxtruHelper;
    auto const &vertices   = shell.GetPolygon().GetVertices();
    Precision const *x_cpu = vertices.x();
    Precision const *y_cpu = vertices.y();
    const auto nvertices   = vertices.size();
    Precision *x_gpu_ptr   = AllocateOnGpu<Precision>(nvertices * sizeof(Precision));
    Precision *y_gpu_ptr   = AllocateOnGpu<Precision>(nvertices * sizeof(Precision));
    vecgeom::CopyToGpu(x_cpu, x_gpu_ptr, sizeof(Precision) * nvertices);
    vecgeom::CopyToGpu(y_cpu, y_gpu_ptr, sizeof(Precision) * nvertices);

    auto final_gpu_ptr = CopyToGpuImpl<UnplacedExtruded>(in_gpu_ptr, static_cast<int>(nvertices), x_gpu_ptr, y_gpu_ptr,
                                                         shell.GetLowerZ(), shell.GetUpperZ(), fGlobalConvexity);

    FreeFromGpu(x_gpu_ptr);
    FreeFromGpu(y_gpu_ptr);
    return final_gpu_ptr;
  }

  auto const &runtime = GetStruct().fTslRuntimeHelper;
  VECGEOM_VALIDATE(runtime.fNFacets > 0 && runtime.fFacets != nullptr && runtime.fBVH != nullptr,
                   << "Attempted to copy UnplacedExtruded without initialized tessellated runtime data");

  auto gpu_bvh_ptr = cuda::AllocateDeviceBVHBuffer<float>(1);
  runtime.fBVH->CopyToGpu(gpu_bvh_ptr);

  size_t nfacets      = runtime.fNFacets;
  auto gpu_facets_ptr = AllocateOnGpu<cuda::TriangularTile<double>>(sizeof(TriangularTile<double>) * nfacets);
  vecgeom::CopyToGpu((char *)runtime.fFacets, (char *)gpu_facets_ptr, sizeof(TriangularTile<double>) * nfacets);

  auto final_gpu_ptr =
      CopyToGpuImpl<UnplacedExtruded>(in_gpu_ptr, nfacets, gpu_facets_ptr, gpu_bvh_ptr, fGlobalConvexity);

  const auto *this_bytes                  = reinterpret_cast<const char *>(this);
  const auto *runtime_bytes               = reinterpret_cast<const char *>(&runtime);
  const std::size_t runtime_struct_offset = static_cast<std::size_t>(runtime_bytes - this_bytes);
  // Preserve the device facet/BVH pointers set by the GPU constructor; copy only host-computed metadata.
  constexpr std::size_t first_pointer_offset_in_runtime = offsetof(TessellatedRuntimeStruct<Precision>, fFacets);
  vecgeom::CopyToGpu((char *)this + runtime_struct_offset, (char *)(in_gpu_ptr.GetPtr()) + runtime_struct_offset,
                     first_pointer_offset_in_runtime);

  return final_gpu_ptr;
}

DevicePtr<cuda::VUnplacedVolume> UnplacedExtruded::CopyToGpu() const { return CopyToGpuImpl<UnplacedExtruded>(); }

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedExtruded>::SizeOf();
template void DevicePtr<cuda::UnplacedExtruded>::Construct() const;
template void DevicePtr<cuda::UnplacedExtruded>::Construct(size_t, TriangularTile<double> *, BVH<float> *, bool) const;
template void DevicePtr<cuda::UnplacedExtruded>::Construct(int, Precision *, Precision *, Precision, Precision,
                                                           bool) const;

} // namespace cxx

#endif

} // namespace vecgeom
