#ifndef VECGEOM_SURFACE_BUILDER_H
#define VECGEOM_SURFACE_BUILDER_H

#include <VecGeom/surfaces/base/CpuTypes.h>

namespace vgbrep {

namespace builder {

template <typename Real_t>
UnplacedSurface CreateUnplacedSurface(SurfaceType type, Real_t *data = nullptr, bool flip = false)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  switch (type) {
  case kPlanar:
    return UnplacedSurface(type);
  case kCylindrical:
  case kSpherical:
    cpudata.fCylSphData.push_back({data[0], flip});
    return UnplacedSurface(type, cpudata.fCylSphData.size() - 1);
  case kConical:
    cpudata.fConeData.push_back({data[0], data[1], flip});
    return UnplacedSurface(type, cpudata.fConeData.size() - 1);
  case kTorus:
  case kGenSecondOrder:
    std::cout << "kTorus, kGenSecondOrder unhandled\n";
    return UnplacedSurface(type);
  };
  return UnplacedSurface(type);
}

// Creators for different types of frames.
template <typename Real_t>
Frame CreateFrame(FrameType type, WindowMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fWindowMasks.size();
  cpudata.fWindowMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
Frame CreateFrame(FrameType type, RingMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fRingMasks.size();
  cpudata.fRingMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
Frame CreateFrame(FrameType type, ZPhiMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fZPhiMasks.size();
  cpudata.fZPhiMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
Frame CreateFrame(FrameType type, QuadrilateralMask<Real_t> const &mask)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fQuadMasks.size();
  cpudata.fQuadMasks.push_back(mask);
  return Frame(type, id);
}

template <typename Real_t>
int CreateLocalTransformation(Transformation const &trans)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fLocalTrans.size();
  cpudata.fLocalTrans.push_back(trans);
  return id;
}

template <typename Real_t>
int CreateLocalSurface(UnplacedSurface const &unplaced, Frame const &frame, int trans, bool use_surf_safety)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  int id        = cpudata.fLocalSurfaces.size();
  cpudata.fLocalSurfaces.push_back({unplaced, frame, trans, use_surf_safety});
  return id;
}

template <typename Real_t>
int AddSurfaceToShell(int logical_id, int isurf)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  if (cpudata.fShells.size() == 0) {
    std::cout << "BrepHelper::AddSurfaceToShell: need to call SetNvolumes first\n";
    return -1;
  }
  assert(logical_id < (int)cpudata.fShells.size() && "surface shell id exceeding number of volumes");
  int id = cpudata.fShells[logical_id].fSurfaces.size();
  cpudata.fShells[logical_id].fSurfaces.push_back(isurf);
  return id;
}

template <typename Real_t>
void AddLogicToShell(int logical_id, LogicExpressionCPU &logic)
{
  // Add solid logic to existing shell logic
  auto &cpudata  = CPUsurfData<Real_t>::Instance();
  auto &crtlogic = cpudata.fShells[logical_id].fLogic;
  crtlogic.insert(crtlogic.end(), logic.begin(), logic.end());
}

} // namespace builder

} // namespace vgbrep

#endif
