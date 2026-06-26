/// \file BVH_V2.cpp
/// \brief Host/device transfer support for BVH_V2 (see base/BVH_V2.h).

#include "VecGeom/base/BVH_V2.h"

#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Interface.h"
#include <stdexcept>
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_CUDA_INTERFACE
template <typename Real_t>
DevicePtr<cuda::BVH_V2<Real_t>> BVH_V2<Real_t>::CopyToGpu(void *addr) const
{
  // Top-level node type so the device pointer can be named where cuda::BVH_V2 is only forward-declared.
  using HostNode_t = BVH_V2Node<Real_t>;

  if (!addr) throw std::logic_error("Cannot copy BVH_V2 into a null pointer!");

  cuda::BVH_V2Node<Real_t> *dNodes;
  int *dPrimId;
  cuda::AABB<Real_t> *dAABBs;

  // BVH_V2 ships three flat arrays (vs. five for BVH): the packed node array (bounds + child index + count),
  // the primitive-id permutation, and the per-primitive AABBs. All are POD and copied as raw bytes.
  VECGEOM_DEVICE_API_CALL(Malloc((void **)&dNodes, fNNodes * sizeof(HostNode_t)));
  VECGEOM_DEVICE_API_CALL(Malloc((void **)&dPrimId, fNPrim * sizeof(int)));
  VECGEOM_DEVICE_API_CALL(Malloc((void **)&dAABBs, fNPrim * sizeof(AABB<Real_t>)));

  VECGEOM_DEVICE_API_CALL(Memcpy((void *)dNodes, (void *)fNodes, fNNodes * sizeof(HostNode_t),
                                 VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
  VECGEOM_DEVICE_API_CALL(Memcpy((void *)dPrimId, (void *)fPrimId, fNPrim * sizeof(int),
                                 VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));
  VECGEOM_DEVICE_API_CALL(Memcpy((void *)dAABBs, (void *)fAABBs, fNPrim * sizeof(AABB<Real_t>),
                                 VECGEOM_DEVICE_API_SYMBOL(MemcpyHostToDevice)));

  DevicePtr<cuda::BVH_V2<Real_t>> dBVH(addr);
  dBVH.Construct(fRootId, fNPrim, fNNodes, fMaxDepth, dNodes, dPrimId, dAABBs);
  return dBVH;
}

template DevicePtr<cuda::BVH_V2<float>> BVH_V2<float>::CopyToGpu(void *) const;
#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA
namespace cxx {

template void DevicePtr<cuda::BVH_V2<float>>::Construct(int, int, int, int, cuda::BVH_V2Node<float> *, int *,
                                                        cuda::AABB<float> *) const;

} // namespace cxx
#endif

} // namespace vecgeom
