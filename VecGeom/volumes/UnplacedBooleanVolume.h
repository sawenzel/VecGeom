#ifndef UNPLACEDBOOLEANVOLUME_H_
#define UNPLACEDBOOLEANVOLUME_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/BooleanStruct.h"
#include "VecGeom/volumes/kernel/BooleanImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"
#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedMultiUnion.h"
#endif

#include <string>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_1v(class, UnplacedBooleanVolume, BooleanOperation, Arg1);

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
namespace BooleanHelper {

VECCORE_ATT_HOST_DEVICE
BooleanStruct const *GetBooleanStruct(VUnplacedVolume const *unplaced);

bool ContainsHalfSpace(VUnplacedVolume const *unplaced);

bool HasRootUnsupportedHalfSpaceUnion(VUnplacedVolume const *unplaced);

bool IsFiniteBody(VPlacedVolume const *placed);

bool IsFiniteBody(VUnplacedVolume const *unplaced);

VECCORE_ATT_HOST_DEVICE
size_t CountBooleanNodes(VUnplacedVolume const *unplaced, size_t &nunion, size_t &nintersection, size_t &nsubtraction);

UnplacedMultiUnion *Flatten(VUnplacedVolume const *unplaced, size_t min_unions = 3,
                            Transformation3D const *trbase = nullptr, UnplacedMultiUnion *munion = nullptr);
} // namespace BooleanHelper
#endif

/**
 * A class representing a simple UNPLACED boolean volume A-B
 * It takes two template arguments:
 * 1.: the mother (or left) volume A in unplaced form
 * 2.: the (or right) volume B in placed form, acting on A with a boolean operation;
 * the placement is with respect to the left volume
 *
 *
 *
 * will be a boolean solid where two boxes are subtracted
 * and B is only translated (not rotated) with respect to A
 *
 */
template <BooleanOperation Op>
class UnplacedBooleanVolume : public UnplacedVolumeImplHelper<BooleanImplementation<Op>>, public AlignedBase {

public:
  BooleanStruct fBoolean;
  using UnplacedVolumeImplHelper<BooleanImplementation<Op>>::fGlobalConvexity;

  // the constructor
  VECCORE_ATT_HOST_DEVICE
  UnplacedBooleanVolume(BooleanOperation op, VPlacedVolume const *left, VPlacedVolume const *right)
      : fBoolean(op, left, right)
  {
    fGlobalConvexity = false;
#ifndef VECCORE_CUDA
    VUnplacedVolume::ComputeBBox();
    if (fBoolean.fLeftVolume->IsAssembly() || fBoolean.fRightVolume->IsAssembly()) {
      throw std::runtime_error("Trying to make boolean out of assembly which is not supported\n");
    }
#else
    VUnplacedVolume::ComputeBBox();
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual bool IsBoolean() const override { return true; }

  VECCORE_ATT_HOST_DEVICE
  virtual ESolidType GetType() const override { return ESolidType::boolean; }

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedBooleanVolume<Op>>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  BooleanOperation GetOp() const { return fBoolean.fOp; }

  VECCORE_ATT_HOST_DEVICE
  BooleanStruct const &GetStruct() const { return fBoolean; }

  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  Precision Capacity() const override
  {
    if (fBoolean.fCapacity < 0.) {
      fBoolean.fCapacity = VUnplacedVolume::EstimateCapacity(1000000);
    }
    return fBoolean.fCapacity;
  }

  Precision SurfaceArea() const override
  {
    if (fBoolean.fSurfaceArea < 0.) {
      fBoolean.fSurfaceArea = VUnplacedVolume::EstimateSurfaceArea(1000000);
    }
    return fBoolean.fSurfaceArea;
  }

  VECCORE_ATT_HOST_DEVICE
  void Extent(Vector3D<Precision> &aMin, Vector3D<Precision> &aMax) const override;

  Vector3D<Precision> SamplePointOnSurface() const override;

  std::string GetEntityType() const { return "BooleanVolume"; }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const override
  {
#ifndef VECCORE_CUDA
    auto op            = this->GetOp();
    const char *opname = (op == kUnion ? "Union" : (op == kIntersection ? "Intersection" : "Subtraction"));
    printf("UnplacedBooleanVol<%s>{fLeft=[%i]<%s>, fRight=[%i]<%s>}", opname, this->GetLeft()->id(),
           this->GetLeft()->GetName(), this->GetRight()->id(), this->GetRight()->GetName());
#else
    printf("UnplacedBooleanVol<%i>{fLeft=[%i], fRight=[%i]}", this->GetOp(), this->GetLeft()->id(),
           this->GetRight()->id());
#endif
    printf("\n\t----------------------------------------");
    printf("\n\tfLeft:  [%i] ", this->GetLeft()->id());
    this->GetLeft()->GetUnplacedVolume()->Print();
    printf("\n\tfRight: [%i] ", this->GetRight()->id());
    this->GetRight()->GetUnplacedVolume()->Print();
    printf("\n\t----------------------------------------\n");
  }

  virtual void Print(std::ostream &os) const override
  {
#ifndef VECCORE_CUDA
    auto op = this->GetOp();
    os << "UnplacedBooleanVol<" << (op == kUnion ? "Union" : (op == kIntersection ? "Intersection" : "Subtraction"))
       << ">{fLeft=<" << this->GetLeft()->id() << ">, fRight=<" << this->GetRight()->id() << ">}\n";
#endif
    os << "fLeft:  [" << this->GetLeft()->id() << "] ";
    this->GetLeft()->GetUnplacedVolume()->Print(os);
    os << "fRight: [" << this->GetRight()->id() << "] ";
    this->GetRight()->GetUnplacedVolume()->Print(os);
    os << "\n";
  }

  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *GetLeft() const { return fBoolean.fLeftVolume; }
  VECCORE_ATT_HOST_DEVICE
  VPlacedVolume const *GetRight() const { return fBoolean.fRightVolume; }

#ifndef VECCORE_CUDA
  /** @brief Count number of Boolean nodes of different types under this Boolean volume */
  VECCORE_ATT_HOST_DEVICE
  size_t CountNodes(size_t &nunion, size_t &nintersection, size_t &nsubtraction) const
  {
    return BooleanHelper::CountBooleanNodes(this, nunion, nintersection, nsubtraction);
  }

  /** @brief Flatten the Boolean node structure and create multiple union volumes if possible */
  UnplacedMultiUnion *Flatten(size_t min_unions = 1, Transformation3D const *trbase = nullptr,
                              UnplacedMultiUnion *munion = nullptr)
  {
    return BooleanHelper::Flatten(this, min_unions, trbase, munion);
  }
#endif

private:
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id, const int copy_no, const int child_id,
#endif
                                           VPlacedVolume *const placement = NULL) const override;

  void SetLeft(VPlacedVolume const *pvol) { fBoolean.fLeftVolume = pvol; }
  void SetRight(VPlacedVolume const *pvol) { fBoolean.fRightVolume = pvol; }

  friend class vecgeom::GeoManager;
}; // End class

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif /* UNPLACEDBOOLEANVOLUME_H_ */
