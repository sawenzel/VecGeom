
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
  struct Safetyval {
    vecgeom::Vector3D<Real_t> onsurf; ///< point on surface
    Real_t safety{0.};                ///< computed safety value
    short int isurf{-1};              ///< surface to which it applies (-1 = un-initialized)
    char op{0};                       ///< operator applying to it: 0 = uninitialized, 1 = '&', -1 = '|'

    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    void Reset()
    {
      safety = 0.;
      isurf  = -1;
      op     = 0;
    }

    ///< @brief Perform the reduction with another safety value
    VECCORE_ATT_HOST_DEVICE
    VECGEOM_FORCE_INLINE
    void SwapReduction(const Safetyval &other)
    {
      // make sure the operator is reset at exit
      auto crt_op = op;
      op          = 0;
      // In case one of the surfaces is not defined, act as identity
      if (other.isurf < 0) return;
      if (isurf < 0) {
        operator=(other);
        return;
      }
      assert(crt_op != 0);

      if ((crt_op > 0) ^ (safety > other.safety)) {
        // swap current safety with other one
        safety = other.safety;
        onsurf = other.onsurf;
        isurf  = other.isurf;
      }
    }
  };

  ///< Lambda to get the safety for individual framed surfaces of the same logical volume
  auto safetySurf = [&](Safetyval &safety_surf) {
    // Convert point from volume to local surface coordinates
    auto trans             = surfdata.fLocalSurf[safety_surf.isurf].fTrans;
    Vector3D<Real_t> local = trans.Transform(plocalVol);
    auto const &framedsurf = surfdata.fLocalSurf[safety_surf.isurf];
    bool flipped           = framedsurf.fLogicId < 0;
    auto const &unplaced   = framedsurf.fSurface;
    unplaced.Safety(local, /*exiting ^*/ flipped, surfdata, safety_surf.safety, safety_surf.onsurf);
  };

  auto safetyFrame = [&](Safetyval &safety_surf) {
    // Compute safety from point projected on surface to the surface frame
    auto const &framedsurf = surfdata.fLocalSurf[safety_surf.isurf];
    bool valid             = false;
    auto safety            = framedsurf.fFrame.Safety(safety_surf.onsurf, safety_surf.safety, surfdata, valid);
    if (valid) safety_surf.safety = safety;
  };

  // Implementation of the following infix Boolean expression evaluation:
  // * The logic expression contains operands (surface ids), operators `&` or `|` and depths changed by `(` or `)`
  // * Operands may be negated `!` meaning that the corresponding half-space is flipped compared to the standard normal
  // convention. Negation is ignored because it is taken into account in the surface flipping, which negates the safety.
  //   * The logic expression is evaluated left to right taking the following actions depending on the current item:
  //     * Operands trigger signed safety evaluation for the half-space surface, using the convention: outside =
  //     positive.
  //     * Operators are cached for the current depth. Upon reading an operand and having a cached operator, min/max is
  //       called according to the operation, the result replacing the currently cached value. The safety reduction
  //       between two values is dependent on the operator:
  //       * safety(a & b) = max(safety_a, safety_b)
  //       * safety(a | b) = min(safety_a, safety_b)
  //       * if current operand does not have an operator (e.g. first in an expression in a new level), the reduction
  //       is done with a unit operand not changing the current one
  //     * Depth increase `(` pushes to the stack the current computed safety AND operator as sign of the safety: `+`
  //     for &
  //       and `-` for |
  //     * Depth decrease `)` pops the cached safety and operation and performs the safety reduction with the current
  //       cached value.
  // * At the end of the expression evaluation, the sign of the safety is negated if exiting is true, and the safety to
  // the actual closest frame is computed

  // TO DO: assert that the maximum depth is not hit after logic expression simplification
  constexpr int kStackSize = 8; // maximum depth (nested Boolean operations) for the input logic expression.
  Safetyval cached_safety[kStackSize];
  Safetyval crt_safety;

  int depth = 0;
  unsigned i;
  for (i = 0; i < logic.size(); ++i) {
    auto item = logic[i];
    if (item == lplus) {
      // increase depth: cache current safety and reset it
      cached_safety[depth++] = crt_safety;
      crt_safety.Reset();
    } else if (item == lminus) {
      // decrease depth: check if cached safety is valid
      depth--;
      assert(depth >= 0);
      crt_safety.op = cached_safety[depth].op;
      crt_safety.SwapReduction(cached_safety[depth]);
    } else if (item == lnot) {
      // negate = true;
    } else if (LogicExpression::is_operator_token(item)) {
      crt_safety.op = (item == land) ? 1 : -1;
      i++;
    } else {
      // This is a surface index
      Safetyval new_safety;
      new_safety.isurf = short(item);
      safetySurf(new_safety);
      crt_safety.SwapReduction(new_safety);
    }
  }
  assert(depth == 0);
  if (exiting) crt_safety.safety = -crt_safety.safety;
  // If needed, compute safety to the frame
  if (crt_safety.safety > 0. && crt_safety.safety <= safe_max) safetyFrame(crt_safety);
  return crt_safety.safety;
}

} // namespace vgbrep
#endif
