/*
 * volume_utilities.h
 *
 *  Created on: Mar 24, 2014
 *      Author: swenzel
 */

#ifndef VOLUME_UTILITIES_H_
#define VOLUME_UTILITIES_H_

#include "base/Vector3D.h"
#include "base/Global.h"
#include "base/RNG.h"
#include "volumes/PlacedBox.h"

#include <cassert>

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

namespace VECGEOM_NAMESPACE {
namespace volumeUtilities {

using namespace VECGEOM_NAMESPACE;

VECGEOM_INLINE
bool IsHittingVolume(Vector3D<Precision> const &point,
                     Vector3D<Precision> const &dir,
                     VPlacedVolume const &volume) {

#if defined(VECGEOM_ROOT) && defined(VECGEOM_BENCHMARK)
static const TGeoShape * rootshape = volume.ConvertToRoot();
double *safe = NULL;
double rpoint[3];
double rdir[3];
for(int i=0;i<3;i++){
  rpoint[i]=point[i]; rdir[i]=dir[i];}
  return rootshape->DistFromOutside(&rpoint[0], &rdir[0], 3, kInfinity, safe) < 1E20;
#else
   return volume.DistanceToIn(point, dir, kInfinity) < kInfinity;
#endif
}

VECGEOM_INLINE
Vector3D<Precision> SamplePoint(Vector3D<Precision> const &size,
                                const Precision scale = 1) {
  const Vector3D<Precision> ret(
      scale * (1. - 2. * RNG::Instance().uniform()) * size[0],
      scale * (1. - 2. * RNG::Instance().uniform()) * size[1],
      scale * (1. - 2. * RNG::Instance().uniform()) * size[2]
  );
  return ret;
}

VECGEOM_INLINE
Vector3D<Precision> SampleDirection() {

  Vector3D<Precision> dir(
      (1. - 2. * RNG::Instance().uniform()),
      (1. - 2. * RNG::Instance().uniform()),
      (1. - 2. * RNG::Instance().uniform())
  );

  const Precision inverse_norm =
      1. / std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
  dir *= inverse_norm;

  return dir;
}


template<typename TrackContainer>
VECGEOM_INLINE
void FillRandomDirections(TrackContainer &dirs) {
  dirs.resize(dirs.capacity());
  for (int i = 0, iMax = dirs.capacity(); i < iMax; ++i) {
    dirs.set(i, SampleDirection());
  }
}

template<typename TrackContainer>
VECGEOM_INLINE
void FillBiasedDirections(VPlacedVolume const &volume,
                          TrackContainer const &points,
                          const Precision bias,
                          TrackContainer & dirs) {
  assert(bias >= 0. && bias <= 1.);

  const int size = dirs.capacity();
  int n_hits = 0;
  std::vector<bool> hit(size, false);
  int h;

  // Randomize points
  FillRandomDirections(dirs);

  // Check hits
  for (int i = 0; i < size; ++i) {
    for (Vector<Daughter>::const_iterator j = volume.daughters().cbegin();
         j != volume.daughters().cend(); ++j) {
      if (IsHittingVolume(points[i], dirs[i], **j)) {
        n_hits++;
        hit[i] = true;
      }
    }
  }

  // Remove hits until threshold
  while (static_cast<Precision>(n_hits)/static_cast<Precision>(size) >= bias) {
    h = static_cast<int>(
        static_cast<Precision>(size) * RNG::Instance().uniform()
    );
    while (hit[h]) {
      dirs.set(h, SampleDirection());
      for (Vector<Daughter>::const_iterator i = volume.daughters().cbegin(),
           iEnd = volume.daughters().cend(); i != iEnd; ++i) {
        if (!IsHittingVolume(points[h], dirs[h], **i)) {
          n_hits--;
          hit[h] = false;
          break;
        }
      }
    }
  }


  // Add hits until threshold
  while (static_cast<Precision>(n_hits)/static_cast<Precision>(size) < bias) {
    h = static_cast<int>(
        static_cast<Precision>(size) * RNG::Instance().uniform()
    );
    while (!hit[h]) {
      dirs.set(h, SampleDirection());
      for (Vector<Daughter>::const_iterator i = volume.daughters().cbegin(),
           iEnd = volume.daughters().cend(); i != iEnd; ++i) {
        if (IsHittingVolume(points[h], dirs[h], **i)) {
          n_hits++;
          hit[h] = true;
          break;
        }
      }
    }
  }
}

template<typename TrackContainer>
VECGEOM_INLINE
void FillBiasedDirections(LogicalVolume const &volume,
                          TrackContainer const &points,
                          const Precision bias,
                          TrackContainer & dirs)
{
  VPlacedVolume const *const placed = volume.Place();
  FillBiasedDirections(*placed, points, bias, dirs);
  delete placed;
}

template<typename TrackContainer>
VECGEOM_INLINE
void FillUncontainedPoints(VPlacedVolume const &volume,
                           TrackContainer &points) {
  const int size = points.capacity();
  points.resize(points.capacity());
  const Vector3D<Precision> dim = volume.bounding_box()->dimensions();
  for (int i = 0; i < size; ++i) {
    bool contained;
    do {
      points.set(i, SamplePoint(dim));
      contained = false;
      for (Vector<Daughter>::const_iterator j = volume.daughters().cbegin(),
          jEnd = volume.daughters().cend(); j != jEnd; ++j) {
        if ((*j)->Contains( points[i] )) {
          contained = true;
          break;
        }
      }
    } while (contained);
  }
}

template<typename TrackContainer>
VECGEOM_INLINE
void FillContainedPoints(VPlacedVolume const &volume,
                         const double bias,
                         TrackContainer &points,
                         const bool placed = true) {
  const int size = points.capacity();
  points.resize(points.capacity());
  const Vector3D<Precision> dim = volume.bounding_box()->dimensions();
  int insideCount = 0;
  std::vector<bool> insideVector(size, false);
  for (int i = 0; i < size; ++i) {
    points.set(i, SamplePoint(dim));
    for (Vector<Daughter>::const_iterator v = volume.daughters().cbegin(),
         v_end = volume.daughters().cend(); v != v_end; ++v) {
      bool inside = (placed) ? (*v)->Contains(points[i])
                             : (*v)->UnplacedContains(points[i]);
      if (inside) {
        ++insideCount;
        insideVector[i] = true;
      }
    }
  }
  int i = 0;
  while (static_cast<double>(insideCount)/static_cast<double>(size) > bias) {
    while (!insideVector[i]) ++i;
    bool contained = false;
    do {
      points.set(i, SamplePoint(dim));
      for (Vector<Daughter>::const_iterator v = volume.daughters().cbegin(),
           v_end = volume.daughters().end(); v != v_end; ++v) {
        bool inside = (placed) ? (*v)->Contains(points[i])
                               : (*v)->UnplacedContains(points[i]);
        if (inside) {
          contained = true;
          break;
        }
      }
    } while (contained);
    insideVector[i] = false;
    --insideCount;
    ++i;
  }
  i = 0;
  while (static_cast<double>(insideCount)/static_cast<double>(size) < bias) {
    while (insideVector[i]) ++i;
    bool contained = false;
    do {
      const Vector3D<Precision> sample = SamplePoint(dim);
      for (Vector<Daughter>::const_iterator v = volume.daughters().cbegin(),
           v_end = volume.daughters().cend(); v != v_end; ++v) {
        bool inside = (placed) ? (*v)->Contains(sample)
                               : (*v)->UnplacedContains(sample);
        if (inside) {
          points.set(i, sample);
          contained = true;
          break;
        }
      }
    } while (!contained);
    insideVector[i] = true;
    ++insideCount;
    ++i;
  }
}

template<typename TrackContainer>
VECGEOM_INLINE
void FillContainedPoints(VPlacedVolume const &volume,
                         TrackContainer &points,
                         const bool placed = true) {
  FillContainedPoints<TrackContainer>(volume, 1, points, placed);
}

template<typename TrackContainer>
VECGEOM_INLINE
void FillUncontainedPoints(LogicalVolume const &volume,
                           TrackContainer &points) {
  VPlacedVolume const *const placed = volume.Place();
  FillUncontainedPoints(*placed, points);
  delete placed;
}

template <typename TrackContainer>
VECGEOM_INLINE
void FillRandomPoints(VPlacedVolume const &volume,
                      TrackContainer &points) {
  const int size = points.capacity();
  points.resize(points.capacity());
  const Vector3D<Precision> dim = volume.bounding_box()->dimensions();
  for (int i = 0; i < size; ++i) {
    Vector3D<Precision> point;
    do {
      point = SamplePoint(dim);
    } while (!volume.Contains(point));
    points.set(i, point);
  }
}

} // end namespace volumeUtilities
} // end global namespace

#endif /* VOLUME_UTILITIES_H_ */
