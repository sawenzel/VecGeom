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
  using VecInt_t       = std::vector<int>;
  using VecChar_t      = std::vector<char>;
  using MultimapInt_t  = std::multimap<long, int>;
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
  std::vector<TriangleMask_t> fTriangleMasks; 
  std::vector<QuadMask_t> fQuadMasks;         ///< quadrilateral masks
  std::vector<CylData_t> fCylSphData;         ///< data for cyl surfaces
  std::vector<ConeData_t> fConeData;          ///< data for conical surfaces
  std::vector<Transformation> fLocalTrans;    ///< local transformations
  std::vector<Transformation> fGlobalTrans;   ///< global transformations for surfaces in the scene
  std::vector<FramedSurface> fLocalSurfaces;  ///< local surfaces per logical volume
  std::vector<FramedSurface> fFramedSurf;     ///< global surfaces
  std::vector<CommonSurface> fCommonSurfaces; ///< common surfaces
  std::vector<VolumeShellCPU> fShells;        ///< vector of local volume surfaces
  std::vector<VolumeShellCPU> fSceneShells;   ///< vector of scene volume surfaces

  VecInt_t fSceneStartIndex;            ///< Start indices for data indexed by state id (per scene)
  VecInt_t fSceneTouchables;            ///< Number of touchables (per scene)
  std::vector<MultimapInt_t> fSurfHash; ///< maps rotation hash index to a list of common surface id's (per scene)

  std::vector<VecInt_t> fCandidatesEntering; ///< list of entering candidates: scene0...,scene1...
  std::vector<VecInt_t> fCandidatesExiting;  ///< list of exiting candidates: scene0...,scene1...
  std::vector<VecInt_t> fFrameIndEntering; ///< list of start frame indices for entering candidates: scene0...,scene1...
  std::vector<VecInt_t> fFrameIndExiting;  ///< list of start frame indices for exiting candidates: scene0...,scene1...
  std::vector<VecChar_t> fSidesEntering;   ///< list of relevant sides for entering candidates: scene0...,scene1...
  std::vector<VecChar_t> fSidesExiting;    ///< list of relevant sides for exiting candidates: scene0...,scene1...

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
    std::vector<WindowMask_t>().swap(fWindowMasks);
    std::vector<RingMask_t>().swap(fRingMasks);
    std::vector<ZPhiMask_t>().swap(fZPhiMasks);
    std::vector<TriangleMask_t>().swap(fTriangleMasks);
    std::vector<QuadMask_t>().swap(fQuadMasks);
    std::vector<CylData_t>().swap(fCylSphData);
    std::vector<ConeData_t>().swap(fConeData);
    std::vector<Transformation>().swap(fLocalTrans);
    std::vector<Transformation>().swap(fGlobalTrans);
    std::vector<FramedSurface>().swap(fLocalSurfaces);
    std::vector<FramedSurface>().swap(fFramedSurf);
    std::vector<CommonSurface>().swap(fCommonSurfaces);
    std::vector<VolumeShellCPU>().swap(fShells);
    std::vector<VolumeShellCPU>().swap(fSceneShells);
    VecInt_t().swap(fSceneStartIndex);
    VecInt_t().swap(fSceneTouchables);
    std::vector<MultimapInt_t>().swap(fSurfHash);
    std::vector<VecInt_t>().swap(fCandidatesEntering);
    std::vector<VecInt_t>().swap(fCandidatesExiting);
    std::vector<VecInt_t>().swap(fFrameIndEntering);
    std::vector<VecInt_t>().swap(fFrameIndExiting);
    std::vector<VecChar_t>().swap(fSidesEntering);
    std::vector<VecChar_t>().swap(fSidesExiting);
  }

  VecInt_t &GetCandidatesEntering(int scene_id, int state_id)
  {
    return fCandidatesEntering[fSceneStartIndex[scene_id] + state_id];
  }

  VecInt_t &GetCandidatesExiting(int scene_id, int state_id)
  {
    return fCandidatesExiting[fSceneStartIndex[scene_id] + state_id];
  }

  VecInt_t &GetFrameIndEntering(int scene_id, int state_id)
  {
    return fFrameIndEntering[fSceneStartIndex[scene_id] + state_id];
  }

  VecInt_t &GetFrameIndExiting(int scene_id, int state_id)
  {
    return fFrameIndExiting[fSceneStartIndex[scene_id] + state_id];
  }

  VecChar_t &GetSidesEntering(int scene_id, int state_id)
  {
    return fSidesEntering[fSceneStartIndex[scene_id] + state_id];
  }

  VecChar_t &GetSidesExiting(int scene_id, int state_id)
  {
    return fSidesExiting[fSceneStartIndex[scene_id] + state_id];
  }
};

} // namespace vgbrep

#endif
