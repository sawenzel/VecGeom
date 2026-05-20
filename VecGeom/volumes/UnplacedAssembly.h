// LICENSING INFORMATION TBD

#ifndef VECGEOM_UNPLACEDASSEMBLY_H
#define VECGEOM_UNPLACEDASSEMBLY_H

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector.h"
#include "VecGeom/navigation/VNavigator.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/navigation/NavStateFwd.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedAssembly;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedAssembly);

inline namespace VECGEOM_IMPL_NAMESPACE {

// An assembly volume offering navigation interfaces (Contains/Distances/...) in a loose/logical group of volumes
// An UnplacedAssembly is always strongly coupled to a logical volume because the later naturally manages the actual
// group of volumes that define the assembly

// The following construct marks a logical volume as an assembly:

// UnplacedAssembly *ass = new UnplacedAssembly
// LogicalVolume *lv = new LogicalVolume("assembly", ass); // this will implicitly couple ass to lv
// lv->PlacedDaughter(...)

class UnplacedAssembly : public VUnplacedVolume, public AlignedBase {

private:
  // back-reference to the logical volume
  // this gets automatically set upon instantiation of a logical volume with an UnplacedAssembly
  LogicalVolume *fLogicalVolume;

  // caching the extent (bounding box)
  // these members are automatically updated whenever a new volume is added to the assembly
  Vector3D<Precision> fLowerCorner;
  Vector3D<Precision> fUpperCorner;

  void SetLogicalVolume(LogicalVolume *lv) { fLogicalVolume = lv; }
  void UpdateExtent()
  {
    UnplacedAssembly::Extent(fLowerCorner, fUpperCorner);
    ComputeBBox();
  }
  friend class LogicalVolume;

public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedAssembly(); // the constructor

  VECCORE_ATT_HOST_DEVICE
  virtual ~UnplacedAssembly();

  LogicalVolume const *GetLogicalVolume() const { return fLogicalVolume; }

  // add content
  void AddVolume(VPlacedVolume *const);

  // get number of volumes
  size_t GetNVolumes() const { return fLogicalVolume->GetDaughters().size(); }

  // the extent function
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  // Getter to cached bounding box
  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GetLowerCorner() const { return fLowerCorner; }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GetUpperCorner() const { return fUpperCorner; }

  // the ordinary assembly function
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<Precision> const &point) const override
  {
    VECGEOM_ASSERT(fLogicalVolume);
    // check bound box first
    bool inBoundingBox;
    ABBoxImplementation::ABBoxContainsKernel(fLowerCorner, fUpperCorner, point, inBoundingBox);
    if (!inBoundingBox) return false;

    Vector3D<Precision> daughterlocalpoint;
    VPlacedVolume const *nextv;
    return fLogicalVolume->GetLevelLocator()->LevelLocate(fLogicalVolume, point, nextv, daughterlocalpoint);
  }

  VECCORE_ATT_HOST_DEVICE
  EnumInside Inside(Vector3D<Precision> const & /*point*/) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("Assembly inside to be implemented");
#endif
    return static_cast<EnumInside>(EInside::kOutside);
  }

  // an extended contains function needed for navigation
  // if this function returns true it modifies the navigation state to point to the first non-assembly volume
  // the point is contained in
  // this function is not part of the generic UnplacedVolume interface but we could consider doing so
  VECCORE_ATT_HOST_DEVICE
  bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &daughterlocalpoint, NavigationState &state) const
  {
    VECGEOM_ASSERT(fLogicalVolume);
    // check bound box first
    bool inBoundingBox;
    ABBoxImplementation::ABBoxContainsKernel(fLowerCorner, fUpperCorner, point, inBoundingBox);
    if (!inBoundingBox) return false;

    return fLogicalVolume->GetLevelLocator()->LevelLocate(fLogicalVolume, point, state, daughterlocalpoint);
  }

  using VUnplacedVolume::DistanceToOut;
  // DistanceToOut does not make sense -- throw exception
  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const & /*p*/, Vector3D<Precision> const & /*d*/,
                          Precision /*step_max*/ = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("Forbidden DistanceToOut in Assembly called");
#endif
    return -1.;
  }

  VECCORE_ATT_HOST_DEVICE
  Precision DistanceToOut(Vector3D<Precision> const &p, Vector3D<Precision> const &d, Precision step_max,
                          SurfaceHitView<Precision> *hit_info) const override
  {
    if (hit_info) hit_info->Clear();
    return DistanceToOut(p, d, step_max);
  }

  using VUnplacedVolume::SafetyToOut;
  VECCORE_ATT_HOST_DEVICE
  Precision SafetyToOut(Vector3D<Precision> const &) const override
  {
#ifndef VECCORE_CUDA
    throw std::runtime_error("Forbidden SafetyToOut in Assembly called");
#endif
    return -1.;
  }

  // ---------------- SafetyToIn functions -------------------------------------------------------
  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &p) const override
  {
    return fLogicalVolume->GetSafetyEstimator()->ComputeSafetyToDaughtersForLocalPoint(p, fLogicalVolume);
  }

  // ---------------- DistanceToIn functions -----------------------------------------------------

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &p, Vector3D<Precision> const &d,
                                 const Precision /*step_max*/ = kInfLength) const override
  {
    if (!BoxImplementation::Intersect(&fLowerCorner, p, d, 0, kInfLength)) return kInfLength;

    Precision step(kInfLength);
    VPlacedVolume const *pv;
    fLogicalVolume->GetNavigator()->CheckDaughterIntersections(fLogicalVolume, p, d, nullptr, nullptr, step, pv);
    return step;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &p, Vector3D<Precision> const &d, const Precision step_max,
                                 SurfaceHitView<Precision> *hit_info) const override
  {
    if (hit_info) hit_info->Clear();
    return DistanceToIn(p, d, step_max);
  }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &norm) const override { return false; }

  Vector3D<Precision> SamplePointOnSurface() const override;
  Precision Capacity() const override;
  Precision SurfaceArea() const override;

  // some dummy impl for virtual functions
  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override;
  virtual void Print(std::ostream &os) const override;
  virtual int MemorySize() const override { return sizeof(*this); }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedAssembly>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECCORE_CUDA
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           VPlacedVolume *const placement = NULL) const override;
#else
  VECCORE_ATT_DEVICE VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                                      Transformation3D const *const transformation, const int id,
                                                      const int copy_no, const int child_id,
                                                      VPlacedVolume *const placement) const override;
#endif
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_UNPLACEDASSEMBLY_H
