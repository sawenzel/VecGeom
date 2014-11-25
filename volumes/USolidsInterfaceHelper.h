/// \file USolidsInterfaceHelper.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_USOLIDSINTERFACEHELPER_H_
#define VECGEOM_VOLUMES_USOLIDSINTERFACEHELPER_H_


#include "base/Global.h"


#ifndef VECGEOM_USOLIDS

namespace VECGEOM_NAMESPACE {
  struct USolidsInterfaceHelper {
    VECGEOM_CUDA_HEADER_BOTH
    virtual ~USolidsInterfaceHelper() {}
  };
}

#else // Compiling with USolids compatibility

#include "base/Vector3D.h"
#include "volumes/PlacedVolume.h"
#include "UVector3.hh"
#include "VUSolid.hh"

#include <string>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#endif

namespace VECGEOM_NAMESPACE {

/// \brief USolids compatibility signatures.
/// 
/// These do not necessarily provide all the return values promised by the
/// interface, so use volumes in this way with caution.
class USolidsInterfaceHelper : public VUSolid {

public:
 //   VUSolid(const std::string &name);
  USolidsInterfaceHelper(const std::string &name) : VUSolid(name) {}
  USolidsInterfaceHelper() : VUSolid() {}

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(
    Vector3D<Precision> const &position,
    Vector3D<Precision> const &direction,
    Precision stepMax = kInfinity) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual ~USolidsInterfaceHelper() {}

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &position) const =0;

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &position) const =0;

  virtual double DistanceToOut(Vector3D<double> const &point,
                               Vector3D<double> const &direction,
                               Vector3D<double> &normal,
                               bool &convex,
                               double stepMax = kInfinity) const
  {
      assert(0 &&
                 "This DistanceToOut interface was not implemented for this volume.");
          return false;
  }

  virtual double DistanceToOut(Vector3D<double> const &point,
                               Vector3D<double> const &direction,
                               Vector3D<double> &normal,
                               bool &convex) const
  {
    return DistanceToOut(point, direction,normal, convex, kInfinity);
  }

  virtual double SafetyFromOutside(Vector3D<double> const &point,
                                   bool accurate = false) const {
    return SafetyToIn(point);
  }

  virtual double SafetyFromInside(Vector3D<double> const &point,
                                  bool accurate = false) const {
    return SafetyToOut(point);
  }

  virtual bool Normal(Vector3D<double> const &point,
                      Vector3D<double> &normal) const {

    assert(0 &&
           "Normal not implemented for USolids interface compatible volume.");
    return false;
  }

  virtual void Extent(Vector3D<double> &min,
                      Vector3D<double> &max) const {
    assert(0 && "Extent not implemented for USolids interface compatible"
                " volume.");
  }

  virtual std::string GetEntityType() const {
    assert(0 && "GetEntityType not implemented for USolids interface compatible"
                " volume.");
    return std::string();
  }

  virtual double Capacity() {
    assert(0 && "Capacity not implemented for USolids interface compatible"
                " volume.");
    return 0;
  }

  virtual double SurfaceArea() {
    assert(0 && "SurfaceArea not implemented for USolids interface compatible"
                " volume.");
    return 0;
  }

  virtual void GetParametersList(int number, double *array) const {
    assert(0 && "GetParameterList not implemented for USolids interface"
                " compatible volume.");
  }

  virtual VUSolid* Clone() const {
    assert(0 && "Clone not implemented for USolids interface compatible"
                " volume.");
    return NULL;
  }

  virtual std::ostream& StreamInfo(std::ostream &os) const {
    assert(0 && "StreamInfo not implemented for USolids interface compatible"
                " volume.");
    return os;
  }

  virtual UVector3 GetPointOnSurface() const {
    assert(0 && "GetPointOnSurface not implemented for USolids interface"
                " compatible volume.");
    return UVector3();
  }

  virtual void ComputeBBox(UBBox *aBox, bool aStore = false) {
    assert(0 && "ComputeBBox not implemented for USolids interface compatible"
                " volume.");
  }



};

} // End global namespace

#endif // USolids defined

#endif // VECGEOM_VOLUMES_USOLIDSINTERFACEHELPER_H_
