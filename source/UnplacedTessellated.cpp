/// @file UnplacedTessellated.cpp
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#include "VecGeom/volumes/UnplacedTessellated.h"
#include "VecGeom/volumes/SpecializedTessellated.h"
#include "VecGeom/volumes/utilities/GenerationUtilities.h"
#include "VecGeom/base/RNG.h"

#include "VecGeom/management/VolumeFactory.h"
#include <cstddef> // offsetof

namespace vecgeom {

#ifdef VECCORE_CUDA
inline
#endif
    namespace cuda {

// forward declare a function impl in BVHManager.cu
template <typename Real_t>
BVH<Real_t> *AllocateDeviceBVHBuffer(size_t n);
} // namespace cuda

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

size_t UnplacedTessellated::FillFromObjFile(std::string const &objfilename, bool close)
{
  using Vec3      = vecgeom::Vector3D<double>;
  auto parseIndex = [](const std::string &token) -> int {
    // Handles "v", "v/t", "v//n", "v/t/n"
    return std::stoi(token.substr(0, token.find('/'))) - 1;
  };

  std::ifstream in(objfilename);
  if (!in) return 0;

  std::vector<Vec3> vertices;
  std::string line;

  int nfacets = 0;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;

    if (tag == "v") {
      double x, y, z;
      ss >> x >> y >> z;
      vertices.push_back(Vec3(x, y, z));
    } else if (tag == "f") {
      std::string a, b, c;
      ss >> a >> b >> c;

      int i0 = parseIndex(a);
      int i1 = parseIndex(b);
      int i2 = parseIndex(c);

      AddTriangularFacet(vertices[i0], vertices[i1], vertices[i2], true);
      nfacets++;
    }
  }
  if (close) {
    Close();
  }
  return nfacets;
}

UnplacedTessellated *UnplacedTessellated::CreateFromObjFile(std::string const &objfilename, bool close)
{
  auto tsl = new UnplacedTessellated();
  /*auto nfacets = */ tsl->FillFromObjFile(objfilename, close);
  // error handling?
  return tsl;
}

void UnplacedTessellated::Close()
{
  ComputeBBox();
  fTessellated.Close();
  // we can now fill the runtime tessellated struct from fTessellated
  fTessellatedRuntime.InitFrom(fTessellated);
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
  norm       = TessellatedImplementation::NormalKernel<Precision>(GetStruct(), point, valid);
  return valid;
}

#ifdef VECCORE_CUDA
VECCORE_ATT_DEVICE VPlacedVolume *UnplacedTessellated::Create(LogicalVolume const *const logical_volume,
                                                              Transformation3D const *const transformation,
                                                              const int id, const int copy_no, const int child_id,
                                                              VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedTessellated(logical_volume, transformation, id, copy_no, child_id);
    return placement;
  }
  return new SpecializedTessellated(logical_volume, transformation, id, copy_no, child_id);
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
                                                      const int id, const int copy_no, const int child_id,
#endif
                                                      VPlacedVolume *const placement) const
{

  return VolumeFactory::CreateByTransformation<UnplacedTessellated>(volume, transformation,
#ifdef VECCORE_CUDA
                                                                    id, copy_no, child_id,
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
  // we need essentially: the constructed BVH, the container of triangles, and some other data from
  // tessellatedruntimestruct

  // (a) copy the bvh
  auto gpu_bvh_ptr = cuda::AllocateDeviceBVHBuffer<float>(1);
  this->GetStruct().fBVH->CopyToGpu(gpu_bvh_ptr);

  // (b) copy the triangles
  size_t nfacets      = this->GetStruct().fNFacets;
  auto gpu_facets_ptr = AllocateOnGpu<cuda::TriangularTile<double>>(sizeof(TriangularTile<double>) * nfacets);
  vecgeom::CopyToGpu((char *)this->GetStruct().fFacets, (char *)gpu_facets_ptr,
                     sizeof(TriangularTile<double>) * nfacets);

  // construct the instance on the GPU using these 2 pointer data
  auto final_gpu_ptr = CopyToGpuImpl<UnplacedTessellated>(in_gpu_ptr, nfacets, gpu_facets_ptr, gpu_bvh_ptr);

  // (c) finally copy the non-pointer data of TessellatedRuntimestruct (do not want to initialize on GPU)
  const auto *this_bytes              = reinterpret_cast<const char *>(this);
  const auto *tsl_bytes               = reinterpret_cast<const char *>(&fTessellatedRuntime);
  const std::size_t tsl_struct_offset = static_cast<std::size_t>(tsl_bytes - this_bytes);
  // Copy only the non-pointer prefix so the GPU facet/BVH pointers set by Construct() stay intact.
  constexpr std::size_t first_pointer_offset_in_tsl =
      offsetof(decltype(UnplacedTessellated::fTessellatedRuntime), fFacets);
  vecgeom::CopyToGpu((char *)this + tsl_struct_offset, (char *)(in_gpu_ptr.GetPtr()) + tsl_struct_offset,
                     first_pointer_offset_in_tsl);

  return final_gpu_ptr;
}

DevicePtr<cuda::VUnplacedVolume> UnplacedTessellated::CopyToGpu() const { return CopyToGpuImpl<UnplacedTessellated>(); }

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedTessellated>::SizeOf();
template void DevicePtr<cuda::UnplacedTessellated>::Construct() const;
template void DevicePtr<cuda::UnplacedTessellated>::Construct(size_t, TriangularTile<double> *, BVH<float> *) const;

template void ConstructManyOnGpu<vecgeom::cuda::SpecializedVolImplHelper<vecgeom::cuda::TessellatedImplementation>>(
    unsigned long, vecgeom::cxx::DevicePtr<vecgeom::cuda::VPlacedVolume> const *,
    vecgeom::cxx::DevicePtr<vecgeom::cuda::LogicalVolume> const *,
    vecgeom::cxx::DevicePtr<vecgeom::cuda::Transformation3D> const *, unsigned int const *, int const *, int const *);

} // namespace cxx

#endif

} // namespace vecgeom
