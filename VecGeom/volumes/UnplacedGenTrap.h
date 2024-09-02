/// \file UnplacedGenTrap.h
/// \author: swenzel
///  Modified and completed: mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_
#define VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_

#include "VecGeom/base/Cuda.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/volumes/GenTrapStruct.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/SecondOrderSurfaceShell.h"
#include "VecGeom/volumes/kernel/GenTrapImplementation.h"
#include "VecGeom/volumes/UnplacedVolumeImplHelper.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedGenTrap;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedGenTrap);

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A generic trap:
 * see TGeoArb8 or UGenericTrap
 */
class UnplacedGenTrap : public UnplacedVolumeImplHelper<GenTrapImplementation>, public AlignedBase {

public:
  using Vertex_t = Vector3D<Precision>;

  GenTrapStruct<Precision> fGenTrap; /** The generic trapezoid structure */

public:
  /** @brief UnplacedGenTrap dummy constructor */
  VECCORE_ATT_HOST_DEVICE
  UnplacedGenTrap() : fGenTrap() {}

  /** @brief UnplacedGenTrap constructor
   * @param verticesx X positions of vertices in array form
   * @param verticesy Y positions of vertices in array form
   * @param halfzheight The half-height of the GenTrap
   */
  VECCORE_ATT_HOST_DEVICE
  UnplacedGenTrap(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
      : fGenTrap(verticesx, verticesy, halfzheight)
  {
    fGlobalConvexity = !fGenTrap.fIsTwisted;
    ComputeBBox();
  }

  /** @brief UnplacedGenTrap destructor */
  VECCORE_ATT_HOST_DEVICE
  virtual ~UnplacedGenTrap() {}

  VECCORE_ATT_HOST_DEVICE
  bool Initialize(const Precision verticesx[], const Precision verticesy[], Precision halfzheight)
  {
    return fGenTrap.Initialize(verticesx, verticesy, halfzheight);
  }

  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  virtual ESolidType GetType() const override { return ESolidType::gentrap; }

  /** @brief Getter for the generic trapezoid structure */
  VECCORE_ATT_HOST_DEVICE
  GenTrapStruct<Precision> const &GetStruct() const { return fGenTrap; }

  /** @brief Getter for the surface shell */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  SecondOrderSurfaceShell<4> const &GetShell() const { return (fGenTrap.fSurfaceShell); }

  /** @brief Getter for the half-height */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetDZ() const { return (fGenTrap.fDz); }

  /** @brief Setter for the half-height */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  void SetDZ(Precision dz) { fGenTrap.fDz = dz; }

  /** @brief Getter for the twist angle of a face */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Precision GetTwist(int i) const { return (fGenTrap.fTwist[i]); }

  /** @brief Getter for one of the 8 vertices in Vector3D<Precision> form */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vertex_t const &GetVertex(int i) const { return fGenTrap.fVertices[i]; }

  /** @brief Getter for the array of X coordinates of vertices */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const Precision *GetVerticesX() const { return fGenTrap.fVerticesX; }

  /** @brief Getter for the array of Y coordinates of vertices */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const Precision *GetVerticesY() const { return fGenTrap.fVerticesY; }

  /** @brief Getter for the list of vertices */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  const Vertex_t *GetVertices() const { return fGenTrap.fVertices; }

  /** @brief Computes if this gentrap is twisted */
  VECCORE_ATT_HOST_DEVICE
  bool ComputeIsTwisted() { return fGenTrap.ComputeIsTwisted(); }

  /** @brief Computes if the top and bottom quadrilaterals are convex (mandatory) */
  VECCORE_ATT_HOST_DEVICE
  bool ComputeIsConvexQuadrilaterals() { return fGenTrap.ComputeIsConvexQuadrilaterals(); }

  /** @brief Getter for the planarity of lateral surfaces */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsPlanar() const { return (!fGenTrap.fIsTwisted); }

  /** @brief Getter for the global convexity of the trapezoid */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsDegenerated(int i) const { return (fGenTrap.fDegenerated[i]); }

  /** @brief Computes if opposite segments are crossing, making a malformed shape */
  // This can become a general utility
  VECCORE_ATT_HOST_DEVICE
  bool SegmentsCrossing(Vertex_t pa, Vertex_t pb, Vertex_t pc, Vertex_t pd) const
  {
    return fGenTrap.SegmentsCrossing(pa, pb, pc, pd);
  }

  /** @brief Computes and sets the bounding box dimensions/origin */
  VECCORE_ATT_HOST_DEVICE
  void ComputeBoundingBox() { fGenTrap.ComputeBoundingBox(); }

  /** @brief Memory size in bytes */
  virtual int MemorySize() const final { return sizeof(*this); }

  /** @brief Print parameters of the trapezoid */
  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final { fGenTrap.Print(); }

  /** @brief Print parameters of the trapezoid to stream */
  virtual void Print(std::ostream &os) const final;

#ifndef VECCORE_CUDA
  virtual SolidMesh *CreateMesh3D(Transformation3D const &trans, size_t nSegments) const override;
#endif

#ifdef VECGEOM_CUDA_INTERFACE
  /** @brief Size of object on the device */
  size_t DeviceSizeOf() const final { return DevicePtr<cuda::UnplacedGenTrap>::SizeOf(); }
  /** @brief Copy to GPU interface function */
  DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const final;
  /** @brief Copy to GPU implementation */
  DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const final;
#endif

  /** @brief Interface method for computing capacity */
  Precision Capacity() const override { return volume(); }

  /** @brief Implementation of capacity computation */
  Precision volume() const;

  /** @brief Implementation of surface area computation */
  Precision SurfaceArea() const override;

  /** @brief Compute normal vector to surface */
  VECCORE_ATT_HOST_DEVICE
  bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override;

  /** @brief Computes the extent on X/Y/Z of the trapezoid */
  VECCORE_ATT_HOST_DEVICE
  void Extent(Vertex_t &amin, Vertex_t &amax) const override { return fGenTrap.Extent(amin, amax); }

  /** @brief Generates randomly a point on the surface */
  Vertex_t SamplePointOnSurface() const override;

  /** @brief Get type name */
  std::string GetEntityType() const { return "GenTrap"; }

  /** @brief Templated factory for creating a placed volume */
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id, const int copy_no, const int child_id,
#endif
                               VPlacedVolume *const placement = NULL);

  /** @brief Stream trapezoid information in the Geant4 style */
  std::ostream &StreamInfo(std::ostream &os) const;

private:
  /** @brief Factory for specializing the volume */
  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                           const int id, const int copy_no, const int child_id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

}; // end of class declaration
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_
