/// @file UnplacedExtruded.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDEXTRUDED_H_
#define VECGEOM_VOLUMES_UNPLACEDEXTRUDED_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "ExtrudedStruct.h"
#include "VecGeom/volumes/kernel/ExtrudedImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedExtruded;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedExtruded);

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedExtruded : public UnplacedVolumeImplHelper<ExtrudedImplementation>, public AlignedBase {

  // template <typename U>
  // using vector_t = vecgeom::Vector<U>;
  template <typename U>
  using vector_t = std::vector<U>;

private:
  ExtrudedStruct fXtru; ///< Structure storing the data for the tessellated solid

public:
  /** @brief Dummy constructor */
  VECCORE_ATT_HOST_DEVICE
  UnplacedExtruded() : fXtru() { ComputeBBox(); }

  /** @brief Constructor providing polygon vertices and sections */
  UnplacedExtruded(int nvertices, XtruVertex2 const *vertices, int nsections, XtruSection const *sections)
      : fXtru(nvertices, vertices, nsections, sections)
  {
    fGlobalConvexity = (nsections == 2) && fXtru.IsConvexPolygon();
    ComputeBBox();
  }

  UnplacedExtruded(int nvertices, const Precision *x, const Precision *y, Precision zmin, Precision zmax)
      : fXtru(nvertices, x, y, zmin, zmax)
  {
    fGlobalConvexity = fXtru.IsConvexPolygon();
    ComputeBBox();
  }

  VECCORE_ATT_HOST_DEVICE
  UnplacedExtruded(size_t ntriangles, TriangularTile<double> *triangle_ptr, BVH<float> *bvh, bool globalConvexity)
      : fXtru()
  {
    // CUDA copies use the compact tessellated runtime helper; host section data is not needed on device.
    fXtru.fInitialized      = true;
    fXtru.fIsSxtru          = false;
    fXtru.fTslRuntimeHelper = TessellatedRuntimeStruct<Precision>(ntriangles, triangle_ptr, bvh);
    fGlobalConvexity        = globalConvexity;
  }

  VECCORE_ATT_HOST_DEVICE
  UnplacedExtruded(int nvertices, Precision *x, Precision *y, Precision lowerz, Precision upperz, bool globalConvexity)
      : fXtru()
  {
    // Preserve the host SExtru dispatch on CUDA copies so safeties use the same helper.
    fXtru.fInitialized = true;
    fXtru.fIsSxtru     = true;
    fXtru.fSxtruHelper.Init(nvertices, x, y, lowerz, upperz);
    fGlobalConvexity = globalConvexity;
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::extruded; }

  VECCORE_ATT_HOST_DEVICE
  ExtrudedStruct const &GetStruct() const { return fXtru; }

  /** @brief Initialize */
  void Initialize(int nvertices, XtruVertex2 const *vertices, int nsections, XtruSection const *sections)
  {
    fXtru.Initialize(nvertices, vertices, nsections, sections);
    fGlobalConvexity = (nsections == 2) && fXtru.IsConvexPolygon();
  }

  /** @brief GetThe number of sections */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNSections() const { return fXtru.GetNSections(); }

  /** @brief Get section i */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  XtruSection GetSection(int i) const { return fXtru.GetSection(i); }

  /** @brief Get the number of vertices */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  size_t GetNVertices() const { return fXtru.GetNVertices(); }

  auto GetMeshHelper() const { return fXtru.GetMeshHelper(); }

  /** @brief Get the polygon vertex i */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void GetVertex(int i, Precision &x, Precision &y) const { fXtru.GetVertex(i, x, y); }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override;

  // Computes capacity of the shape in [length^3]
  Precision Capacity() const override;

  // VECCORE_ATT_HOST_DEVICE
  Precision SurfaceArea() const override;

  int ChooseSurface() const;

  Vector3D<Precision> SamplePointOnSurface() const override;

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  virtual void Print(std::ostream &os) const final;

  virtual int memory_size() const final { return sizeof(*this); }
  std::string GetEntityType() const { return "Extruded"; }

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);
#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedExtruded>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  std::ostream &StreamInfo(std::ostream &os) const;

#ifndef VECCORE_CUDA
#ifdef VECGEOM_ROOT
  TGeoShape const *ConvertToRoot(char const *label) const;
#endif

#ifdef VECGEOM_GEANT4
  G4VSolid const *ConvertToGeant4(char const *label) const;
#endif
#endif

private:
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id, const int copy_no, const int child_id,
#endif
                                           VPlacedVolume *const placement = NULL) const override; // final;
};

template <>
struct Maker<UnplacedExtruded> {
  template <typename... ArgTypes>
  static UnplacedExtruded *MakeInstance(const size_t nvertices, XtruVertex2 const *vertices, const int nsections,
                                        XtruSection const *sections);
};

using GenericUnplacedExtruded = UnplacedExtruded;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDEXTRUDED_H_
