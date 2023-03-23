#ifndef VECGEOM_SURFACE_CPUTYPES_H
#define VECGEOM_SURFACE_CPUTYPES_H

#include <VecGeom/surfaces/base/CommonTypes.h>

namespace vgbrep {

using LogicExpressionCPU = std::vector<logic_int>;

// Placeholder (on host) for all surfaces belonging to a volume. An array of those will be indexed
// by the logical volume id. Also an intermediate helper for building portals.
// Note: the local surfaces defined by solids will have local references that will be changed by
// the flattening process, depending on the scene on which the parent volume will be flattened
struct VolumeShellCPU {
  std::vector<int> fSurfaces; ///< Local surface id's for this volume
  LogicExpressionCPU fLogic;  ///< Logic expression for the solid
};

// Surface data used only on CPU during the conversion process
template <typename Real_t>
struct CPUsurfData {
  using SurfData_t     = SurfData<Real_t>;
  using CylData_t      = CylData<Real_t>;
  using ConeData_t     = ConeData<Real_t>;
  using SphData_t      = SphData<Real_t>;
  using WindowMask_t   = WindowMask<Real_t>;
  using RingMask_t     = RingMask<Real_t>;
  using ZPhiMask_t     = ZPhiMask<Real_t>;
  using TriangleMask_t = TriangleMask<Real_t>;
  using QuadMask_t     = QuadrilateralMask<Real_t>;

  std::vector<WindowMask_t> fWindowMasks;     ///< rectangular masks
  std::vector<RingMask_t> fRingMasks;         ///< ring masks
  std::vector<ZPhiMask_t> fZPhiMasks;         ///< cylindrical masks
  std::vector<QuadMask_t> fQuadMasks;         ///< quadrilateral masks
  std::vector<CylData_t> fCylSphData;         ///< data for cyl surfaces
  std::vector<ConeData_t> fConeData;          ///< data for conical surfaces
  std::vector<Transformation> fLocalTrans;    ///< local transformations
  std::vector<Transformation> fGlobalTrans;   ///< global transformations for surfaces in the scene
  std::vector<FramedSurface> fLocalSurfaces;  ///< local surfaces
  std::vector<FramedSurface> fFramedSurf;     ///< global surfaces
  std::vector<CommonSurface> fCommonSurfaces; ///< common surfaces
  std::vector<VolumeShellCPU> fShells;        ///< vector of local volume surfaces
  std::vector<std::vector<int>> fCandidates;  ///< candidate lists for each state
  std::vector<std::vector<int>> fFrameInd;    ///< start frame index per candidate
  std::multimap<int, int> fSurfHash;          ///< maps rotation hash index to a list of common surface id's

private:
  CPUsurfData() = default;

public:
  static VECGEOM_FORCE_INLINE CPUsurfData<Real_t> &Instance()
  {
    static CPUsurfData<Real_t> gCPUsurfdata;
    return gCPUsurfdata;
  }

  void Clear()
  {
    // Dispose of surface data and shrink the container
    fWindowMasks.clear();
    std::vector<WindowMask_t>().swap(fWindowMasks);
    fRingMasks.clear();
    std::vector<RingMask_t>().swap(fRingMasks);
    fZPhiMasks.clear();
    std::vector<ZPhiMask_t>().swap(fZPhiMasks);
    fQuadMasks.clear();
    std::vector<QuadMask_t>().swap(fQuadMasks);
    fCylSphData.clear();
    std::vector<CylData_t>().swap(fCylSphData);
    fConeData.clear();
    std::vector<ConeData_t>().swap(fConeData);
    fLocalTrans.clear();
    std::vector<Transformation>().swap(fLocalTrans);
    fGlobalTrans.clear();
    std::vector<Transformation>().swap(fGlobalTrans);
    fLocalSurfaces.clear();
    std::vector<FramedSurface>().swap(fLocalSurfaces);
    fFramedSurf.clear();
    std::vector<FramedSurface>().swap(fFramedSurf);
    fCommonSurfaces.clear();
    std::vector<CommonSurface>().swap(fCommonSurfaces);
    for (size_t i = 0; i < fShells.size(); ++i) {
      fShells[i].fSurfaces.clear();
      std::vector<int>().swap(fShells[i].fSurfaces);
    }
    fShells.clear();
    std::vector<VolumeShellCPU>().swap(fShells);
  }
};

} // namespace vgbrep

#endif
