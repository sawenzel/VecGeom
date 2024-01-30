#ifndef VECGEOM_SURFACE_SOLIDCONVERTER_H_
#define VECGEOM_SURFACE_SOLIDCONVERTER_H_

#include <VecGeom/surfaces/Model.h>

#include <VecGeom/surfaces/conv/BoxConverter.h>
#include <VecGeom/surfaces/conv/TubeConverter.h>
#include <VecGeom/surfaces/conv/CutTubeConverter.h>
#include <VecGeom/surfaces/conv/ParallelepipedConverter.h>
#include <VecGeom/surfaces/conv/ConeConverter.h>
#include <VecGeom/surfaces/conv/PolyconeConverter.h>
#include <VecGeom/surfaces/conv/SExtrudedConverter.h>
#include <VecGeom/surfaces/conv/TrdConverter.h>
#include <VecGeom/surfaces/conv/TrapezoidConverter.h>
#include <VecGeom/surfaces/conv/PolyhedronConverter.h>
#include <VecGeom/surfaces/conv/BooleanConverter.h>
#include <VecGeom/surfaces/conv/ScaledConverter.h>
#include <VecGeom/volumes/ScaledShape.h>

namespace vgbrep {
namespace conv {

/// @brief Dispacher to different solid converters
/// @tparam Real_t Precision type
/// @param solid Unplaced solid to be converted
/// @param volId Logical volume id
/// @param localtrans Local transformation, used only for Boolean volumes
/// @return Conversion success
template <typename Real_t>
bool CreateSolidSurfaces(vecgeom::VUnplacedVolume const *solid, int volId, Transformation *localtrans = nullptr)
{
  bool success{true};
  auto &cpudata     = CPUsurfData<Real_t>::Instance();
  auto const &shell = cpudata.fShells[volId];
  auto isurf_first  = shell.fSurfaces.size();

  auto createSurfacesLocal = [](vecgeom::VUnplacedVolume const *solid, int volId) {
    auto box = dynamic_cast<vecgeom::UnplacedBox const *>(solid);
    if (box) return conv::CreateBoxSurfaces<Real_t>(*box, volId);

    auto tube = dynamic_cast<vecgeom::UnplacedTube const *>(solid);
    if (tube) return conv::CreateTubeSurfaces<Real_t>(*tube, volId);

    auto cuttube = dynamic_cast<vecgeom::UnplacedCutTube const *>(solid);
    if (cuttube) return conv::CreateTubeSurfaces<Real_t>(*cuttube, volId);

    auto para = dynamic_cast<vecgeom::UnplacedParallelepiped const *>(solid);
    if (para) return conv::CreateParallelepipedSurfaces<Real_t>(*para, volId);

    auto cone = dynamic_cast<vecgeom::UnplacedCone const *>(solid);
    if (cone) return conv::CreateConeSurfaces<Real_t>(*cone, volId);

    auto polycone = dynamic_cast<vecgeom::UnplacedPolycone const *>(solid);
    if (polycone) return conv::CreatePolyconeSurfaces<Real_t>(*polycone, volId);

    auto xtru = dynamic_cast<vecgeom::UnplacedSExtruVolume const *>(solid);
    if (xtru) return conv::CreateSExtrudedSurfaces<Real_t>(*xtru, volId);

    auto trd = dynamic_cast<vecgeom::UnplacedTrd const *>(solid);
    if (trd) return conv::CreateTrdSurfaces<Real_t>(*trd, volId);

    auto trap = dynamic_cast<vecgeom::UnplacedTrapezoid const *>(solid);
    if (trap) return conv::CreateTrapezoidSurfaces<Real_t>(*trap, volId);

    auto polyhedron = dynamic_cast<vecgeom::UnplacedPolyhedron const *>(solid);
    if (polyhedron) return conv::CreatePolyhedronSurfaces<Real_t>(*polyhedron, volId);

    auto bstruct = vecgeom::BooleanHelper::GetBooleanStruct(solid);
    if (bstruct) return conv::CreateBooleanSurfaces<Real_t>(*bstruct, volId);

    auto scaled = dynamic_cast<vecgeom::UnplacedScaledShape const *>(solid);
    if (scaled) return conv::CreateScaledSurfaces<Real_t>(*scaled, volId);

    return false;
  };

  success = createSurfacesLocal(solid, volId);

  // If there is a local transformation, apply it to all surfaces
  if (success && localtrans) {
    auto isurf_last = shell.fSurfaces.size();
    for (size_t i = isurf_first; i < isurf_last; ++i) {
      auto const &surf = cpudata.fLocalSurfaces[shell.fSurfaces[i]];
      Transformation trans(*localtrans);
      trans.MultiplyFromRight(cpudata.fLocalTrans[surf.fTrans]);
      cpudata.fLocalTrans[surf.fTrans] = trans;
    }
  }
  return success;
}

/// @brief Converter for scaled solids
/// @tparam Real_t Precision type
/// @param scaled Scaled solid to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateScaledSurfaces(vecgeom::UnplacedScaledShape const &scaled, int logical_id)
{
  auto const &vec_scale = scaled.GetScale().Scale();
  if (!ApproxEqualVector(vec_scale, vecgeom::Vector3D<Real_t>{1, 1, -1})) {
    VECGEOM_LOG(critical) << "UnplacedScaledShape having scale " << vec_scale << " not supported";
    return false;
  }
  auto success = CreateSolidSurfaces<Real_t>(scaled.UnscaledShape(), logical_id);
  // Reflect all framed surfaces held by the shell
  auto &cpudata     = CPUsurfData<Real_t>::Instance();
  auto const &shell = cpudata.fShells[logical_id];
  for (int lsurf_id : shell.fSurfaces) {
    FramedSurface const &lsurf = cpudata.fLocalSurfaces[lsurf_id];
    auto const &trans          = cpudata.fLocalTrans[lsurf.fTrans];
    // Reflect the framed surface
    Transformation scalez(0, 0, 0, 0, 0, 0, 1, 1, -1);
    Transformation refl_trans         = trans * scalez;
    cpudata.fLocalTrans[lsurf.fTrans] = refl_trans;
  }
  return success;
}

/// @brief Converter for Boolean solids
/// @tparam Real_t Precision type
/// @param bstruct Boolean structure to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateBooleanSurfaces(vecgeom::BooleanStruct const &bstruct, int logical_id)
{
  Transformation trans;
  AppendLogicTo<Real_t>(bstruct, trans, logical_id);

  auto &cpudata = CPUsurfData<Real_t>::Instance();
  // Finalize logic expression
  auto &crtlogic = cpudata.fShells[logical_id].fLogic;
  vgbrep::logichelper::LogicExpressionConstruct lc(crtlogic);
  lc.Simplify(crtlogic);

  // Assign logic id to each surface
  for (auto id : cpudata.fShells[logical_id].fSurfaces) {
    // All frames of Boolean surfaces must be checked
    cpudata.fLocalSurfaces[id].fUseSurfSafety = false;
    // Set the logic id for Boolean surfaces
    if (logichelper::is_negated(id, crtlogic))
      cpudata.fLocalSurfaces[id].fLogicId = -id;
    else
      cpudata.fLocalSurfaces[id].fLogicId = id;
  }

  vgbrep::logichelper::insert_jumps(crtlogic);

  return true;
}

template <typename Real_t>
void AppendLogicTo(vecgeom::BooleanStruct const &bstruct, Transformation const &trans, int logical_id)
{
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  auto &logic   = cpudata.fShells[logical_id].fLogic;
  // left node
  // open parenthesis
  logic.push_back(lplus);
  vecgeom::Transformation3D tr_left(trans);
  tr_left.MultiplyFromRight(*bstruct.fLeftVolume->GetTransformation());
  auto const unplaced_left = bstruct.fLeftVolume->GetUnplacedVolume();
  auto bstruct_left        = vecgeom::BooleanHelper::GetBooleanStruct(unplaced_left);
  if (bstruct_left)
    AppendLogicTo<Real_t>(*bstruct_left, tr_left, logical_id);
  else
    CreateSolidSurfaces<Real_t>(unplaced_left, logical_id, &tr_left);
  // close parenthesis
  logic.push_back(lminus);

  // operator
  switch (bstruct.fOp) {
  case vecgeom::kUnion:
    logic.push_back(lor);
    break;
  case vecgeom::kIntersection:
    logic.push_back(land);
    break;
  case vecgeom::kSubtraction:
    logic.push_back(land);
    logic.push_back(lnot);
    break;
  };

  // right node
  // open parenthesis
  logic.push_back(lplus);
  vecgeom::Transformation3D tr_right(trans);
  tr_right.MultiplyFromRight(*bstruct.fRightVolume->GetTransformation());
  auto const unplaced_right = bstruct.fRightVolume->GetUnplacedVolume();
  auto bstruct_right        = vecgeom::BooleanHelper::GetBooleanStruct(unplaced_right);
  if (bstruct_right)
    AppendLogicTo<Real_t>(*bstruct_right, tr_right, logical_id);
  else
    CreateSolidSurfaces<Real_t>(unplaced_right, logical_id, &tr_right);
  // close parenthesis
  logic.push_back(lminus);
}

} // namespace conv
} // namespace vgbrep
#endif
