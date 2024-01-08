#ifndef VECGEOM_SURFACE_SOLIDCONVERTER_H_
#define VECGEOM_SURFACE_SOLIDCONVERTER_H_

#include <VecGeom/surfaces/Model.h>

#include <VecGeom/surfaces/conv/BoxConverter.h>
#include <VecGeom/surfaces/conv/TubeConverter.h>
#include <VecGeom/surfaces/conv/ParallelepipedConverter.h>
#include <VecGeom/surfaces/conv/ConeConverter.h>
#include <VecGeom/surfaces/conv/SExtrudedConverter.h>
#include <VecGeom/surfaces/conv/TrdConverter.h>
#include <VecGeom/surfaces/conv/PolyhedronConverter.h>
#include <VecGeom/surfaces/conv/BooleanConverter.h>

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

  auto createSurfacesLocal = [&]() {
    auto box = dynamic_cast<vecgeom::UnplacedBox const *>(solid);
    if (box) return conv::CreateBoxSurfaces<Real_t>(*box, volId);

    auto tube = dynamic_cast<vecgeom::UnplacedTube const *>(solid);
    if (tube) return conv::CreateTubeSurfaces<Real_t>(*tube, volId);

    auto para = dynamic_cast<vecgeom::UnplacedParallelepiped const *>(solid);
    if (para) return conv::CreateParallelepipedSurfaces<Real_t>(*para, volId);

    auto cone = dynamic_cast<vecgeom::UnplacedCone const *>(solid);
    if (cone) return conv::CreateConeSurfaces<Real_t>(*cone, volId);

    auto xtru = dynamic_cast<vecgeom::UnplacedSExtruVolume const *>(solid);
    if (xtru) return conv::CreateSExtrudedSurfaces<Real_t>(*xtru, volId);

    auto trd = dynamic_cast<vecgeom::UnplacedTrd const *>(solid);
    if (trd) return conv::CreateTrdSurfaces<Real_t>(*trd, volId);

    auto polyhedron = dynamic_cast<vecgeom::UnplacedPolyhedron const *>(solid);
    if (polyhedron) return conv::CreatePolyhedronSurfaces<Real_t>(*polyhedron, volId);

    auto bstruct = vecgeom::BooleanHelper::GetBooleanStruct(solid);
    if (bstruct) return conv::CreateBooleanSurfaces<Real_t>(*bstruct, volId);

    return false;
  };

  success = createSurfacesLocal();

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

/// @brief Converter for Boolean solids
/// @tparam Real_t Precision type
/// @param bstruct Boolean structure to be converted
/// @param logical_id Id of the logical volume
/// @return Conversion success
template <typename Real_t>
bool CreateBooleanSurfaces(vecgeom::BooleanStruct const &bstruct, int logical_id)
{
  using Bnode   = logichelper::Bnode;
  auto &cpudata = CPUsurfData<Real_t>::Instance();
  Transformation trans;
  Bnode top(trans, logical_id, bstruct);
  AppendLogicTo<Real_t>(top, cpudata.fShells[logical_id].fLogic);
  // Assign logic id to each surface
  for (auto id : cpudata.fShells[logical_id].fSurfaces) {
    // All frames of Boolean surfaces must be checked
    cpudata.fLocalSurfaces[id].fUseSurfSafety = false;
    // Set the logic id for Boolean surfaces
    if (logichelper::is_negated(id, cpudata.fShells[logical_id].fLogic))
      cpudata.fLocalSurfaces[id].fLogicId = -id;
    else
      cpudata.fLocalSurfaces[id].fLogicId = id;
  }
  return true;
}

template <typename Real_t>
void AppendLogicTo(logichelper::Bnode const &node, LogicExpressionCPU &logic, bool negate)
{
  using Bnode  = logichelper::Bnode;
  using Placed = logichelper::Placed;
  // open paranthesys
  logic.push_back(lplus);

  auto left_complexity  = node.left_->GetComplexity();
  auto right_complexity = node.right_->GetComplexity();
  // use commutativity of AND/OR to keep left node as least complex
  bool swap = left_complexity > right_complexity;
  if (swap) {
    auto complexity  = left_complexity;
    left_complexity  = right_complexity;
    right_complexity = complexity;
  }
  Placed *new_left   = swap ? node.right_ : node.left_;
  Placed *new_right  = swap ? node.left_ : node.right_;
  bool new_neg_left  = swap ? node.neg_right_ : node.neg_left_;
  bool new_neg_right = swap ? node.neg_left_ : node.neg_right_;

  // left node
  if (left_complexity > 0) {
    auto left_node = static_cast<Bnode *>(new_left);
    AppendLogicTo<Real_t>(*left_node, logic, negate ^ new_neg_left);
  } else {
    // Leaf node. Check if negated.
    // append leaf expression
    logic.push_back(lplus);
    size_t istart = logic.size();
    CreateSolidSurfaces<Real_t>(new_left->fUnplaced, new_left->fVolId, &new_left->fTrans);
    int inserts = 0;
    if (new_neg_left ^ negate) logichelper::negate_logic(logic, istart, logic.size() - 1, inserts);
    logic.push_back(lminus);
  }

  switch (node.op_) {
  case vecgeom::kUnion:
    if (negate)
      logic.push_back(land);
    else
      logic.push_back(lor);
    break;
  case vecgeom::kIntersection:
    if (negate)
      logic.push_back(lor);
    else
      logic.push_back(land);
    break;
  case vecgeom::kSubtraction:
    printf("cannot find subtraction here\n");
    break;
  };

  // right node
  if (right_complexity > 0) {
    auto right_node = static_cast<Bnode *>(new_right);
    AppendLogicTo<Real_t>(*right_node, logic, negate ^ new_neg_right);
  } else {
    // Leaf node. Check if negated.
    // append leaf expression
    logic.push_back(lplus);
    size_t istart = logic.size();
    CreateSolidSurfaces<Real_t>(new_right->fUnplaced, new_right->fVolId, &new_right->fTrans);
    int inserts = 0;
    if (new_neg_right ^ negate) logichelper::negate_logic(logic, istart, logic.size() - 1, inserts);
    logic.push_back(lminus);
  }

  logic.push_back(lminus);
}

} // namespace conv
} // namespace vgbrep
#endif
