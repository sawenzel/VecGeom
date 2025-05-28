/*
 * VNavigator.h
 *
 *  Created on: 17.09.2015
 *      Author: swenzel
 */

#ifndef NAVIGATION_VNAVIGATOR_H_
#define NAVIGATION_VNAVIGATOR_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/VSafetyEstimator.h"
#include "VecGeom/navigation/NavStateFwd.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// some forward declarations
template <typename T>
class Vector3D;
// class NavigationState;
class LogicalVolume;
class Transformation3D;
class VPlacedVolume;

//! base class defining basic interface for navigation ( hit-detection )
//! sub classes implement optimized algorithms for logical volumes
class VNavigator {

public:
  VNavigator() : fSafetyEstimator(nullptr) {}

  VECCORE_ATT_HOST_DEVICE
  VSafetyEstimator const *GetSafetyEstimator() const { return fSafetyEstimator; }

  //! computes the step (distance) to the next object in the geometry hierarchy obtained
  //! by propagating with step along the ray
  //! the next object could be after a boundary and does not necessarily coincide with the object
  //! hit by the ray

  //! this methods transforms the global coordinates into local ones usually calls more specialized methods
  //! like the hit detection on local coordinates
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndPropagatedState(Vector3D<Precision> const & /*globalpoint*/,
                                                  Vector3D<Precision> const & /*globaldir*/,
                                                  Precision /*(physics) step limit */,
                                                  NavigationState const & /*in_state*/,
                                                  NavigationState & /*out_state*/) const = 0;

  //! computes the step (distance) to the next object in the geometry hierarchy obtained
  //! by propagating with step along the ray
  //! updates out_state to contain information about the next hitting boundary:
  //!   - if a daugher is hit: out_state.Top() will be daughter
  //!   - if ray leaves volume: out_state.Top() will point to current volume
  //!   - if step limit > step: out_state == in_state
  //!
  //! This function is essentialy equal to ComputeStepAndPropagatedState without
  //! the relocation part
  virtual Precision ComputeStep(Vector3D<Precision> const & /*globalpoint*/, Vector3D<Precision> const & /*globaldir*/,
                                Precision /*(physics) step limit */, NavigationState const & /*in_state*/,
                                NavigationState & /*out_state*/) const = 0;

  //! as above ... also returns the safety ... does not give_back an out_state
  //! but the in_state might be modified to contain the next daughter when
  //! user specifies indicateDaughterHit = true
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafety(Vector3D<Precision> const & /*globalpoint*/,
                                         Vector3D<Precision> const & /*globaldir*/, Precision /*(physics) step limit */,
                                         NavigationState & /*in_state*/, bool /*calcsafety*/, Precision & /*safety*/,
                                         bool indicateDaughterHit = false) const = 0;

  // an alias interface ( using TGeo name )
  VECCORE_ATT_HOST_DEVICE
  void FindNextBoundaryAndStep(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                               NavigationState const &in_state, NavigationState &out_state, Precision step_limit,
                               Precision &step) const
  {
    step = ComputeStepAndPropagatedState(globalpoint, globaldir, step_limit, in_state, out_state);
  }

  // an alias interface ( using TGeo name )
  void FindNextBoundaryAndStepAndSafety(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                                        NavigationState const &in_state, NavigationState &out_state,
                                        Precision step_limit, Precision &step, bool calcsafety, Precision &safety) const
  {
    step = ComputeStepAndSafetyAndPropagatedState(globalpoint, globaldir, step_limit, in_state, out_state, calcsafety,
                                                  safety);
  }

  // a similar interface, in addition also returning the safety as a result
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafetyAndPropagatedState(Vector3D<Precision> const & /*globalpoint*/,
                                                           Vector3D<Precision> const & /*globaldir*/,
                                                           Precision /*(physics) step limit */,
                                                           NavigationState const & /*in_state*/,
                                                           NavigationState & /*out_state*/, bool /*calcsafefty*/,
                                                           Precision & /*safety_out*/) const = 0;

  // the bool return type indicates if out_state was already modified; this may happen in assemblies;
  // in this case we don't need to copy the in_state to the out state later on
  // NavigationState a pointer since we might want to pass nullptr
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const * /*lvol*/, Vector3D<Precision> const & /*localpoint*/,
                                          Vector3D<Precision> const & /*localdir*/,
                                          NavigationState const * /*in_state*/, NavigationState * /*out_state*/,
                                          Precision & /*step*/, VPlacedVolume const *& /*hitcandidate*/) const = 0;

  /// check if a ray given by localpoint, localdir intersects with any daughter. Possibility
  /// to pass a volume which is blocked/should be ignored in the query. Updates the step as well as the hitcandidate
  /// volume. (This version is useful for G4; assemblies not supported)
  VECCORE_ATT_HOST_DEVICE
  virtual bool CheckDaughterIntersections(LogicalVolume const * /*lvol*/, Vector3D<Precision> const & /*localpoint*/,
                                          Vector3D<Precision> const & /*localdir*/, VPlacedVolume const * /*blocked*/,
                                          Precision & /*step*/, VPlacedVolume const *& /*hitcandidate*/) const
  {
    VECGEOM_ASSERT(false); // Not implemented --- notify of failure !!
    return false;
  }

protected:
  // a common relocate method ( to calculate propagated states after the boundary )
  VECCORE_ATT_HOST_DEVICE
  virtual void Relocate(Vector3D<Precision> const & /*localpoint*/, NavigationState const &__restrict__ /*in_state*/,
                        NavigationState &__restrict__ /*out_state*/) const = 0;

  // a common function to be used by all navigators to ensure consistency in transporting points
  // after a boundary
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Vector3D<Precision> MovePointAfterBoundary(Vector3D<Precision> const &localpoint,
                                                    Vector3D<Precision> const &dir, Precision step)
  {
    const Precision extra = 1E-6; // TODO: to be revisited (potentially going for a more relative approach)
    return localpoint + (step + extra) * dir;
  }

public:
  VECCORE_ATT_DEVICE
  virtual ~VNavigator(){};

  // get name of implementing class
  virtual const char *GetName() const = 0;

  typedef VSafetyEstimator SafetyEstimator_t;

protected:
  VECCORE_ATT_HOST_DEVICE
  VNavigator(VSafetyEstimator *s) : fSafetyEstimator(s) {}
  VSafetyEstimator *fSafetyEstimator; // a pointer to the safetyEstimator which can be used by the Navigator

  // some common code to prepare the outstate
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  static Precision PrepareOutState(NavigationState const &__restrict__ in_state,
                                   NavigationState &__restrict__ out_state, Precision geom_step, Precision step_limit,
                                   VPlacedVolume const *hitcandidate, bool &doneafterthisstep)
  {
    // now we have the candidates and we prepare the out_state
    in_state.CopyTo(&out_state);
    doneafterthisstep = false;

    // if the following is the case we are in the wrong volume;
    // assuming that DistanceToIn returns negative number when point is inside
    // do nothing (step=0) and retry one level higher

    // TODO: put diagnostic code here ( like in original SimpleNavigator )
    if (geom_step == kInfLength && step_limit > 0.) {
      geom_step = vecgeom::kTolerance;
      out_state.SetBoundaryState(true);
      do {
        out_state.Pop();
      } while (out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      doneafterthisstep = true;
      return geom_step;
    }

    // is geometry further away than physics step?
    // this is a physics step
    if (geom_step > step_limit) {
      // don't need to do anything
      geom_step = step_limit;
      out_state.SetBoundaryState(false);
      return geom_step;
    }

    // otherwise it is a geometry step
    out_state.SetBoundaryState(true);
    out_state.SetLastExited();
    if (hitcandidate) out_state.Push(hitcandidate);

    if (geom_step < 0.) {
      // std::cerr << "WARNING: STEP NEGATIVE; NEXTVOLUME " << nexthitvolume << std::endl;
      // InspectEnvironmentForPointAndDirection( globalpoint, globaldir, currentstate );
      geom_step = 0.;
    }
    return geom_step;
  }

  // kernel to be used with both scalar and vector types
  template <typename T>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static T TreatDistanceToMother(VPlacedVolume const *pvol,
                                                                              Vector3D<T> const &localpoint,
                                                                              Vector3D<T> const &localdir, T step_limit)
  {
    T step;
    VECGEOM_VALIDATE(pvol != nullptr, << "currentvolume is null in navigation");
    step = pvol->DistanceToOut(localpoint, localdir, step_limit);
    vecCore::MaskedAssign(step, step < T(0.), T(0.));
    return step;
  }

  // default static function doing the global to local transformation
  // may be redefined in concrete implementations ( for instance in cases where we know the form of the global matrix
  // a-priori )
  // input
  // TODO: think about how we can have scalar + SIMD version
  // note: the last argument is a trick to pass information across function calls ( only exploited in specialized
  // navigators )
  template <typename T>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DoGlobalToLocalTransformation(
      NavigationState const &in_state, Vector3D<T> const &globalpoint, Vector3D<T> const &globaldir,
      Vector3D<T> &localpoint, Vector3D<T> &localdir)
  {
    // calculate local point/dir from global point/dir
    Transformation3D m;
    in_state.TopMatrix(m);
    localpoint = m.Transform(globalpoint);
    localdir   = m.TransformDirection(globaldir);
  }
};

//! template class providing a standard implementation for
//! some interfaces in VNavigator (using the CRT pattern)
template <typename Impl, bool MotherIsConvex = false>
class VNavigatorHelper : public VNavigator {
protected:
  using VNavigator::VNavigator;

public:
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndPropagatedState(Vector3D<Precision> const &globalpoint,
                                                  Vector3D<Precision> const &globaldir, Precision step_limit,
                                                  NavigationState const &in_state,
                                                  NavigationState &out_state) const override
  {
#ifdef DEBUGNAV
    static size_t counter = 0;
    counter++;
#endif
    // calculate local point/dir from global point/dir
    // call the static function for this provided/specialized by the Impl
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();
    Precision step                    = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
    // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
    if (lvol->GetDaughters().size() > 0)
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);

    // fix state
    bool done;
    step = Impl::PrepareOutState(in_state, out_state, step, step_limit, hitcandidate, done);
    if (done) {
      if (out_state.Top() != nullptr) {
        VECGEOM_ASSERT(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
      return step;
    }
    // step was physics limited
    if (!out_state.IsOnBoundary()) return step;

    // otherwise if necessary do a relocation
    // try relocation to refine out_state to correct location after the boundary
    ((Impl *)this)->Impl::Relocate(MovePointAfterBoundary(localpoint, localdir, step), in_state, out_state);
    if (out_state.Top() != nullptr) {
      while (out_state.Top()->IsAssembly()) {
        out_state.Pop();
      }
      VECGEOM_ASSERT(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
    }
    return step;
  }

  virtual Precision ComputeStep(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                                Precision step_limit, NavigationState const &in_state,
                                NavigationState &out_state) const override
  {
#ifdef DEBUGNAV
    static size_t counter = 0;
    counter++;
#endif
    // calculate local point/dir from global point/dir
    // call the static function for this provided/specialized by the Impl
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();
    Precision step                    = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
    // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
    if (lvol->GetDaughters().size() > 0)
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);

    // fix state
    bool done;
    step = Impl::PrepareOutState(in_state, out_state, step, step_limit, hitcandidate, done);
    if (done) {
      if (out_state.Top() != nullptr) {
        VECGEOM_ASSERT(!out_state.Top()->GetLogicalVolume()->GetUnplacedVolume()->IsAssembly());
      }
      return step;
    }
    // step was physics limited
    if (!out_state.IsOnBoundary()) return step;

    return step;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafety(Vector3D<Precision> const &globalpoint, Vector3D<Precision> const &globaldir,
                                         Precision step_limit, NavigationState &in_state, bool calcsafety,
                                         Precision &safety, bool indicateDaughterHit = false) const override
  {
// FIXME: combine this kernel and the one for ComputeStep() into one generic function
#ifdef DEBUGNAV
    static size_t counter = 0;
    counter++;
#endif
    // calculate local point/dir from global point/dir
    // call the static function for this provided/specialized by the Impl
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    NavigationState *out_state = nullptr;

    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    // get safety first ( the benefit here is that we reuse the local points )
    using SafetyE_t = typename Impl::SafetyEstimator_t;
    if (calcsafety) {
      // call the appropriate safety Estimator
      safety = ((SafetyE_t *)fSafetyEstimator)->SafetyE_t::ComputeSafetyForLocalPoint(localpoint, in_state.Top());
    }

    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();
    Precision step                    = step_limit;

    // is the next object certainly further away than the safety
    bool safetydone = calcsafety && safety >= step;

    if (!safetydone) {
      step = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      if (lvol->GetDaughters().size() > 0)
        ((Impl *)this)
            ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, out_state, step, hitcandidate);
    }
    if (indicateDaughterHit && hitcandidate) in_state.Push(hitcandidate);
    return Min(step, step_limit);
  }

  // a similar interface also returning the safety
  // TODO: reduce this evident code duplication with ComputeStepAndPropagatedState
  VECCORE_ATT_HOST_DEVICE
  virtual Precision ComputeStepAndSafetyAndPropagatedState(Vector3D<Precision> const &globalpoint,
                                                           Vector3D<Precision> const &globaldir, Precision step_limit,
                                                           NavigationState const &__restrict__ in_state,
                                                           NavigationState &__restrict__ out_state, bool calcsafety,
                                                           Precision &safety_out) const override
  {
    // calculate local point/dir from global point/dir
    Vector3D<Precision> localpoint;
    Vector3D<Precision> localdir;
    Impl::DoGlobalToLocalTransformation(in_state, globalpoint, globaldir, localpoint, localdir);

    // get safety first ( the benefit here is that we reuse the local points )
    using SafetyE_t = typename Impl::SafetyEstimator_t;
    safety_out      = 0.;
    if (calcsafety) {
      // call the appropriate safety Estimator
      safety_out = ((SafetyE_t *)fSafetyEstimator)->SafetyE_t::ComputeSafetyForLocalPoint(localpoint, in_state.Top());
    }

    VPlacedVolume const *hitcandidate = nullptr;
    auto pvol                         = in_state.Top();
    auto lvol                         = pvol->GetLogicalVolume();
    Precision step                    = Impl::TreatDistanceToMother(pvol, localpoint, localdir, step_limit);
    ;
    if (lvol->GetDaughters().size() > 0)
      // "suck in" algorithm from Impl and treat hit detection in local coordinates for daughters
      ((Impl *)this)
          ->Impl::CheckDaughterIntersections(lvol, localpoint, localdir, &in_state, &out_state, step, hitcandidate);

    // fix state
    bool done;
    step = Impl::PrepareOutState(in_state, out_state, step, step_limit, hitcandidate, done);
    if (done) return step;

    // step was physics limited
    if (!out_state.IsOnBoundary()) return step;

    // otherwise if necessary do a relocation
    // try relocation to refine out_state to correct location after the boundary
    ((Impl *)this)->Impl::Relocate(MovePointAfterBoundary(localpoint, localdir, step), in_state, out_state);
    return step;
  }

protected:
  // a common relocate method ( to calculate propagated states after the boundary )
  VECGEOM_FORCE_INLINE
  VECCORE_ATT_HOST_DEVICE
  virtual void Relocate(Vector3D<Precision> const &pointafterboundary, NavigationState const &__restrict__ in_state,
                        NavigationState &__restrict__ out_state) const override
  {
    // this means that we are leaving the mother
    // alternatively we could use nextvolumeindex like before
    if (out_state.Top() == in_state.Top()) {
      GlobalLocator::RelocatePointFromPathForceDifferent(pointafterboundary, out_state);
#ifdef CHECK_RELOCATION_ERRORS
      VECGEOM_VALIDATE(in_state.Distance(out_state) != 0, << " error relocating when leaving ");
#endif
    } else {
      // continue directly further down ( next volume should have been stored in out_state already )
      VPlacedVolume const *nextvol = out_state.Top();
      out_state.Pop();
      GlobalLocator::LocateGlobalPoint(nextvol, nextvol->GetTransformation()->Transform(pointafterboundary), out_state,
                                       false);
#ifdef CHECK_RELOCATION_ERRORS
      VECGEOM_VALIDATE(in_state.Distance(out_state) != 0, << " error relocating when entering ");
#endif
      return;
    }
  }

public:
  static const char *GetClassName() { return Impl::gClassNameString; }

  virtual const char *GetName() const override { return GetClassName(); }
}; // end class VNavigatorHelper
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* NAVIGATION_VNAVIGATOR_H_ */
