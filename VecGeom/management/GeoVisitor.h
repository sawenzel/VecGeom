// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Geometry physical node tree visitors.
/// \file management/GeoVisitor.h
/// \author Split from GeoManager based on implementation by Philippe Canal and Sandro Wenzel

#ifndef VECGEOM_MANAGEMENT_GEOVISITOR_H_
#define VECGEOM_MANAGEMENT_GEOVISITOR_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/navigation/NavStateFwd.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/// A visitor functor interface used when iterating over the geometry tree.
template <typename Container>
class GeoVisitor {
protected:
  Container &c_;

public:
  GeoVisitor(Container &c) : c_(c){};

  virtual void apply(VPlacedVolume *, int level = 0) = 0;
  virtual ~GeoVisitor() {}
};

/// A visitor functor interface used when iterating over the geometry tree.
/// This visitor gets injected path information.
template <typename Container>
class GeoVisitorWithAccessToPath {
protected:
  Container &c_;

public:
  GeoVisitorWithAccessToPath(Container &c) : c_(c){};

  virtual void apply(NavigationState *state, int level = 0) = 0;
  virtual ~GeoVisitorWithAccessToPath() {}
};

/// A visitor functor interface used when iterating over the geometry tree.
/// This visitor gets injected path information, the mother navigation index and the current daughter index
class GeoVisitorNavIndex {

public:
  GeoVisitorNavIndex(){};

  virtual NavIndex_t apply(NavStatePath *state, int level, NavIndex_t mother, int dind, NavIndex_t &id) = 0;
  virtual ~GeoVisitorNavIndex() {}
};

/// A basic implementation of a GeoVisitor.
template <typename Container>
class SimpleLogicalVolumeVisitor : public GeoVisitor<Container> {
public:
  SimpleLogicalVolumeVisitor(Container &c) : GeoVisitor<Container>(c) {}
  virtual void apply(VPlacedVolume *vol, int /*level*/)
  {
    LogicalVolume const *lvol = vol->GetLogicalVolume();
    if (std::find(this->c_.begin(), this->c_.end(), lvol) == this->c_.end()) {
      this->c_.push_back(const_cast<LogicalVolume *>(lvol));
    }
  }
  virtual ~SimpleLogicalVolumeVisitor() {}
};

template <typename Container>
class SimplePlacedVolumeVisitor : public GeoVisitor<Container> {
public:
  SimplePlacedVolumeVisitor(Container &c) : GeoVisitor<Container>(c) {}
  virtual void apply(VPlacedVolume *vol, int /* level */) { this->c_.push_back(vol); }
  virtual ~SimplePlacedVolumeVisitor() {}
};

/// A visitor to find out the geometry depth.
class GetMaxDepthVisitor {
private:
  int maxdepth_;

public:
  GetMaxDepthVisitor() : maxdepth_(0) {}
  void apply(VPlacedVolume * /* vol */, int level) { maxdepth_ = (level > maxdepth_) ? level : maxdepth_; }
  int getMaxDepth() const { return maxdepth_; }
};

/// A visitor to find out the total number of geometry nodes.
class GetTotalNodeCountVisitor {
private:
  int fTotalNodeCount;

public:
  GetTotalNodeCountVisitor() : fTotalNodeCount(0) {}
  void apply(VPlacedVolume *, int /* level */) { fTotalNodeCount++; }
  int GetTotalNodeCount() const { return fTotalNodeCount; }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_MANAGEMENT_GEOVISITOR_H_
