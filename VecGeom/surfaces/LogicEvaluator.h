
#ifndef VECGEOM_SURFACE_LOGICEVALUATOR_H
#define VECGEOM_SURFACE_LOGICEVALUATOR_H

#include <VecGeom/surfaces/Model.h>

namespace vgbrep {

/// @brief Evaluate the inside result of the logic expression.
/// @param plocalVol Point in local volume coordinates
/// @param volId Volume id
/// @param logic Logic expression
/// @param surfdata Surface data storage
/// @param logic_id Logical id entering/exiting surface for which the logic is known
/// @param is_inside whether the known entering/exiting surface is inside or not
/// @return Inside property
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE bool EvaluateInside(vecgeom::Vector3D<Real_t> const &plocalVol, int volId,
                                            LogicExpression const &logic, SurfData<Real_t> const &surfdata,
                                            const int logic_id = vecgeom::kMaximumInt, const bool is_inside = 0)
{
  auto test_bit  = [](unsigned bset, int bit) { return (bset & (unsigned(1) << bit)) > 0; };
  auto set_bit   = [](unsigned &bset, int bit) { bset |= unsigned(1) << bit; };
  auto reset_bit = [](unsigned &bset, int bit) { bset &= ~(unsigned(1) << bit); };
  auto swap_bit  = [](unsigned &bset, int bit) { bset ^= unsigned(1) << bit; };
  ///< Lambda to get the inside for individual unplaced surfaces of the same logical volume
  auto insideSurf = [&](int isurf, bool flip) {
    // Convert point from volume to local surface coordinates
    auto itrans                 = surfdata.fLocalSurf[isurf].fTrans;
    Vector3D<Real_t> plocalSurf = itrans.Transform(plocalVol);
    auto const &unplaced        = surfdata.fLocalSurf[isurf].fSurface;
    return unplaced.Inside(plocalSurf, surfdata, flip);
  };
  unsigned stack  = 0;
  unsigned negate = 0;
  int depth       = 0;
  bool result     = false;
  unsigned i;
  for (i = 0; i < logic.size(); ++i) {
    switch (logic[i]) {
    case lplus:
      depth++;
      break;
    case lminus:
      result = test_bit(stack, depth);
      reset_bit(negate, depth--);
      result ^= test_bit(negate, depth);
      if (result)
        set_bit(stack, depth);
      else
        reset_bit(stack, depth);
      break;
    case lnot:
      swap_bit(negate, depth);
      break;
    case lor:
      if (test_bit(stack, depth))
        i = logic[i + 1] - 1;
      else
        i++;
      break;
    case land:
      if (!test_bit(stack, depth))
        i = logic[i + 1] - 1;
      else
        i++;
      break;
    default:
      // This is an operand
      result = logic[i] == std::abs(logic_id) ? is_inside ^ bool(negate) : insideSurf(int(logic[i]), bool(negate));
      result ^= test_bit(negate, depth);
      reset_bit(negate, depth);
      // We can ignore the previous value because of short-circuiting
      if (result)
        set_bit(stack, depth);
      else
        reset_bit(stack, depth);
    }
  }
  assert(depth == 0);
  return (stack & 1) > 0;
}

/// @brief Evaluate the isotropic safety for the logic expression.
/// @param plocalVol Point in local volume coordinates
/// @param volId Volume id
/// @param logic Logic expression
/// @param surfdata Surface data storage
/// @return Safety value. Infine length if safety larger than safe_max
template <typename Real_t>
VECCORE_ATT_HOST_DEVICE Real_t EvaluateSafety(vecgeom::Vector3D<Real_t> const &plocalVol, int volId, bool exiting,
                                              LogicExpression const &logic, SurfData<Real_t> const &surfdata,
                                              Real_t safe_max = vecgeom::InfinityLength<Real_t>())
{
  ///< Lambda to get the safety for individual framed surfaces of the same logical volume
  Vector3D<Real_t> onsurf_crt;
  auto safetySurf = [&](int isurf, Real_t &safety_surf) {
    // Convert point from volume to local surface coordinates
    auto trans             = surfdata.fLocalSurf[isurf].fTrans;
    Vector3D<Real_t> local = trans.Transform(plocalVol);
    auto const &framedsurf = surfdata.fLocalSurf[isurf];
    bool flipped           = framedsurf.fLogicId < 0;
    auto const &unplaced   = framedsurf.fSurface;

    bool can_compute = unplaced.Safety(local, exiting ^ flipped, surfdata, safety_surf, onsurf_crt);
    return can_compute;
  };

  auto safetyFrame = [&](int isurf, Real_t &safety) {
    // Compute safety from point projected on surface to the surface frame
    auto const &framedsurf = surfdata.fLocalSurf[isurf];
    bool valid             = false;
    safety                 = framedsurf.fFrame.Safety(onsurf_crt, safety, surfdata, valid);
    return valid;
  };

  auto safety_reduction = [](Real_t saf1, Real_t saf2, bool and_exiting_xor) {
    return and_exiting_xor ? vecCore::math::Max(saf1, saf2) : vecCore::math::Min(saf1, saf2);
  };

  // Implementation of the following infix Boolean expression evaluation:
  // - logic expression contains operands (surface ids), operators `&` or `|` and indents `(` or `)`
  // - operands may be negated `!` meaning that the corresponding half-space is flipped compared to the standard normal
  // convention.
  // - the logic expression is evaluated left to right taking the following actions depending on the current item:
  // * operands trigger signed safety evaluation for the surface, considering negation if present. In case the surface
  // is not visible for entering/exiting, the negative safety is considered infinite and affects accordingly the Boolean
  // operation(s) at the current expression indentation (depth).
  // * operators are cached for the current depth. Upon reading an operand and having a cached operator, min/max is
  // called according to the operation, the result replacing the currently cached value.
  // * indent increase `(` pushes to the stack the current computed safety AND operator as sign of the safety: `+` for &
  // and `-` for |
  // * indent decrease `)` pops the cached safety and operation and performs the safety reduction with the current
  // cached value.

  // TO DO: assert that the maximum indenting level is not hit after logic expression simplification
  constexpr int kStackSize = 8; // maximum indenting level (nested Boolean operations) for the input logic expression.
  Real_t cached_safety[kStackSize];
  char cached_op[kStackSize];
  char crt_op    = 0; // No-operator
  bool crt_valid = false;
  // bool negate       = false;
  Real_t crt_safety = vecgeom::InfinityLength<Real_t>();

  int depth = 0;
  unsigned i;
  for (i = 0; i < logic.size(); ++i) {
    auto item = logic[i];
    if (item == lplus) {
      cached_op[depth]       = crt_op;
      cached_safety[depth++] = (crt_valid) ? crt_safety : Real_t(-1);
      crt_op                 = 0;
      crt_valid              = false;
    } else if (item == lminus) {
      // Check if cached safety is valid
      depth--;
      if (cached_safety[depth] > Real_t(0)) {
        if (crt_valid)
          crt_safety = safety_reduction(crt_safety, cached_safety[depth], (cached_op[depth] > 0) ^ exiting);
        else
          crt_safety = cached_safety[depth];
        crt_valid = true;
      }
      assert(depth >= 0);
    } else if (item == lnot) {
      // negate = true;
    } else if (LogicExpression::is_operator_token(item)) {
      crt_op = (item == land) ? 1 : -1;
      i++;
    } else {
      // This is a surface index
      Real_t safety;
      // Compute safety to the unplaced surface
      bool can_compute = safetySurf(int(item), safety);
      bool valid       = can_compute && safety > -vecgeom::kToleranceDist<Real_t> && safety <= safe_max;
      safety           = vecCore::math::Max(safety, Real_t(0));
      // If needed, compute safety to the frame
      if (valid) valid = safetyFrame(int(item), safety);
      if (valid) {
        if (crt_valid && crt_op)
          crt_safety = safety_reduction(crt_safety, safety, (crt_op > 0) ^ exiting);
        else
          crt_safety = safety;
        crt_valid = true;
      }
      // Negation info already considered during safety calculation, so redundant here
      // if (negate) last_value = !last_value;
      // negate = false;
    }
  }
  assert(depth == 0);
  return (crt_valid) ? crt_safety : vecgeom::InfinityLength<Real_t>();
}

} // namespace vgbrep
#endif
