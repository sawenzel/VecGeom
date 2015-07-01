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
#include "volumes/LogicalVolume.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "management/GeoManager.h"
#include <cstdio>
#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif
#include <cassert>

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#endif

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {
namespace volumeUtilities {

/**
 * @brief Is the trajectory through a point along a direction hitting a volume?
 * @details If ROOT is available and VECGEOM_BENCHMARK is set, use
 *    ROOT to calculate it, otherwise use VecGeom utilities.
 * @param point is the starting point
 * @param dir is the direction of the trajectory
 * @param volume is the shape under test
 * @return true/false whether the trajectory hits the volume
 */
VECGEOM_INLINE
bool IsHittingVolume(Vector3D<Precision> const &point,
                     Vector3D<Precision> const &dir,
                     VPlacedVolume const &volume) {
   assert( !volume.Contains(point) );
#if defined(USEROOTFORHITDETECTION)
   std::shared_ptr<TGeoShape const> rootshape(volume.ConvertToRoot());
   Transformation3D const *m = volume.GetTransformation();
   Vector3D<Precision> rpoint = m->Transform(point);
   Vector3D<Precision> rdir = m->TransformDirection(dir);
   return rootshape->DistFromOutside((double*)&rpoint[0], (double*)&rdir[0], 3, vecgeom::kInfinity) < 1E20;
#else
   return volume.DistanceToIn(point, dir, vecgeom::kInfinity) < vecgeom::kInfinity;
#endif
}

// utility function to check if track hits any daughter of input logical volume
inline
bool IsHittingAnyDaughter( Vector3D<Precision> const &point,
                           Vector3D<Precision> const &dir,
                           LogicalVolume const &volume ){
  for (int daughter = 0; daughter < volume.GetDaughters().size(); ++daughter) {
    if (IsHittingVolume(point, dir, *volume.GetDaughters()[daughter])) {
                return true;
    }
  }
  return false;
}

/**
 * @brief Returns a random point, based on a sampling rectangular volume.
 * @details Mostly used for benchmarks and navigation tests
 * @param size is a Vector3D containing the rectangular dimensions of the sampling volume
 * @param scale an optional scale factor (default is 1)
 * @return a random output point
 */
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

/**
 *  @brief Returns a random, normalized direction vector.
 *  @details Mostly used for benchmarks, when a direction is needed.
 *  @return a random, normalized direction vector
 */
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


/**
 *  @brief Fills a container with random normalized directions.
 *  @param dirs is the output container, provided by the caller
 */
template<typename TrackContainer>
VECGEOM_INLINE
void FillRandomDirections(TrackContainer &dirs) {
  dirs.resize(dirs.capacity());
  for (int i = 0, iMax = dirs.capacity(); i < iMax; ++i) {
    dirs.set(i, SampleDirection());
  }
}

/**
 * @brief Fills a container with biased normalized directions.
 * @details Directions are randomly assigned first, and then the
 *    fraction of hits is measured and compared to suggested bias.
 *    Then some directions will be modified as needed, to force the
 *    sample as a whole to have the suggested hit bias (@see bias).
 * @param volume provided must have daughter volumes.  Those daughters
 *    are used to determine the hit bias (@see bias).
 * @param points provided, and not modified.
 * @param bias is a real number in the range [0,1], which suggests the
 *    fraction of points hitting any of the daughter volumes.
 * @param dirs is the output directions container, provided by the
 *    caller.
 */
template<typename TrackContainer>
VECGEOM_INLINE
void FillBiasedDirections(VPlacedVolume const &volume,
                          TrackContainer const &points,
                          Precision bias,
                          TrackContainer & dirs) {
  assert(bias >= 0. && bias <= 1.);

  if( bias>0. && volume.GetDaughters().size()==0 ) {
    printf("\nFillBiasedDirections ERROR:\n bias=%f requested, but no daughter volumes found.\n", bias);
    //// should throw exception, but for now just abort
    // printf("FillBiasedDirections: aborting...\n");
    // exit(1);
    ///== temporary: reset bias to zero
    bias=0.0;
  }

  const int size = dirs.capacity();
  int n_hits = 0;
  std::vector<bool> hit(size, false);

  // Randomize directions
  FillRandomDirections(dirs);

  // Check hits
  for (int track = 0; track < size; ++track) {
    if (IsHittingAnyDaughter(points[track], dirs[track], *volume.GetLogicalVolume())) {
      n_hits++;
      hit[track] = true;
    }
  }

  // Remove hits until threshold
  printf("VolumeUtilities: FillBiasedDirs: nhits/size = %i/%i and requested bias=%f\n", n_hits, size, bias);
  int tries = 0;
  int maxtries = 10000*size;
  while (static_cast<Precision>(n_hits)/static_cast<Precision>(size) > bias) {
    //while (n_hits > 0) {
    tries++;
    if(tries%1000000 == 0) {
      printf("%s line %i: Warning: %i tries to reduce bias... volume=%s. Please check.\n", __FILE__, __LINE__, tries, volume.GetLabel().c_str());
    }

    int track =  static_cast<int>(static_cast<Precision>(size) * RNG::Instance().uniform());
    int internaltries = 0;
    while (hit[track]) {
      dirs.set(track, SampleDirection());
      internaltries++;
      if( ! IsHittingAnyDaughter( points[track], dirs[track], *volume.GetLogicalVolume() ) ){
	  n_hits--;
	  hit[track]=false;
	  //	  tries = 0;
      }
      if(internaltries%1000000 == 0) {
	printf("%s line %i: Warning: %i tries to reduce bias... current bias %d volume=%s. Please check.\n", __FILE__, __LINE__, internaltries, n_hits, volume.GetLabel().c_str());
	// try another track
	break;
      }
    }
  }

  // crosscheck
  {
  int crosscheckhits=0;
  for( int track = 0; track<size; ++track )
    if( IsHittingAnyDaughter( points[track], dirs[track], *volume.GetLogicalVolume() ) ) crosscheckhits++;
  assert( crosscheckhits == n_hits && "problem with hit count == 0");
  }

  // Add hits until threshold
  tries = 0;
  while (static_cast<Precision>(n_hits)/static_cast<Precision>(size) < bias && tries < maxtries) {
    int track = static_cast<int>(
        static_cast<Precision>(size) * RNG::Instance().uniform());
    while (!hit[track] && tries < maxtries) {
      ++tries;
      if (tries%1000000==0) {
        printf("%s line %i: Warning: %i tries to increase bias... volume=%s, current bias=%i/%i=%f.  Please check.\n",
               __FILE__, __LINE__, tries, volume.GetLabel().c_str(), n_hits, size,
               static_cast<Precision>(n_hits)/static_cast<Precision>(size));
      }

      // SW: a potentially much faster algorithm is the following:
      // sample a daughter to hit ( we can adjust the sampling probability according to Capacity or something; then generate point on surface of daughter )
      // set direction accordingly
      uint selecteddaughter = (uint) RNG::Instance().uniform() * volume.GetDaughters().size();
      VPlacedVolume const * daughter = volume.GetDaughters()[selecteddaughter];
      Vector3D<Precision> pointonsurface = daughter->GetPointOnSurface();
      // point is in reference frame of daughter so need to transform it back
      Vector3D<Precision> dirtosurfacepoint = daughter->GetTransformation()->InverseTransform( pointonsurface ) - points[track];
      dirtosurfacepoint.Normalize();
      dirs.set(track, dirtosurfacepoint );

      // the brute force and simple sampling technique is the following
      // dirs.set(h, SampleDirection());
      if(IsHittingAnyDaughter(points[track], dirs[track], *volume.GetLogicalVolume()))
          n_hits++;
          hit[track] = true;
          tries = 0;
      }
  }

  // crosscheck
  {
  int crosscheckhits=0;
  for( int p = 0; p<size; ++p )
    if( IsHittingAnyDaughter( points[p], dirs[p], *volume.GetLogicalVolume() ) ) crosscheckhits++;
  assert( crosscheckhits == n_hits && "problem with hit count");
  }

  if( tries == maxtries )
  {
      printf("WARNING: NUMBER OF DIRECTORY SAMPLING TRIES EXCEEDED MAXIMUM; N_HITS %d; ACHIEVED BIAS %lf \n",n_hits, n_hits/(1.*size));
  }

}

/**
 * @brief Same as previous function, but now taking a LogicalVolume as input.
 * @detail Delegates the filling to the other function (@see FillBiasedDirections).
 */
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

VECGEOM_INLINE
Precision UncontainedCapacity(VPlacedVolume const &volume) {
  Precision momCapacity = const_cast<VPlacedVolume&>(volume).Capacity();
  Precision dauCapacity = 0.;
  unsigned int kk = 0;
  for (Vector<Daughter>::const_iterator j = volume.GetDaughters().cbegin(),
         jEnd = volume.GetDaughters().cend(); j != jEnd; ++j, ++kk) {
    dauCapacity += const_cast<VPlacedVolume*>(*j)->Capacity();
  }
  return momCapacity - dauCapacity;
}

/**
 * @brief Fills the volume with 3D points which are _not_ contained in
 *    any daughters of the input mother volume.
 * @details Requires a proper bounding box from the input volume.
 *    Point coordinates are local to input mother volume.
 * @param volume is the input mother volume containing all output points.
 * @param points is the output container, provided by the caller.
 */
template<typename TrackContainer>
VECGEOM_INLINE
void FillUncontainedPoints(VPlacedVolume const &volume,
                           TrackContainer &points) {
  static double lastUncontCap = 0.0;
  double uncontainedCapacity = UncontainedCapacity(volume);
  if(uncontainedCapacity != lastUncontCap) {
    printf("Uncontained capacity for %s: %g units\n", volume.GetLabel().c_str(), uncontainedCapacity);
    lastUncontCap = uncontainedCapacity;
  }
  if( uncontainedCapacity <= 1000*kTolerance ) {
    std::cout<<"\nVolUtil: FillUncontPts: ERROR: Volume provided <"
             << volume.GetLabel() <<"> does not have uncontained capacity!  Aborting.\n";
    Assert(false);
  }

  const int size = points.capacity();
  points.resize(points.capacity());

  Vector3D<Precision> lower, upper, offset;
  volume.Extent(lower,upper);
  offset = 0.5*(upper+lower);
  const Vector3D<Precision> dim = 0.5*(upper-lower);

  int tries = 0;
  for (int i = 0; i < size; ++i) {
    bool contained;
    Vector3D<Precision> point;
    tries = 0;
    do {
      // ensure that point is contained in mother volume
      do {
        ++tries;
        if(tries%1000000 == 0) {
          printf("%s line %i: Warning: %i tries to find uncontained points... volume=%s.  Please check.\n",
                 __FILE__, __LINE__, tries, volume.GetLabel().c_str());
        }

        point = offset + SamplePoint(dim);
      } while (!volume.UnplacedContains(point));
      points.set(i, point);

      contained = false;
      int kk=0;
      for (Vector<Daughter>::const_iterator j = volume.GetDaughters().cbegin(),
             jEnd = volume.GetDaughters().cend(); j != jEnd; ++j, ++kk) {
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
void FillUncontainedPoints(LogicalVolume const &volume,
                           TrackContainer &points) {
  VPlacedVolume const *const placed = volume.Place();
  FillUncontainedPoints(*placed, points);
  delete placed;
}


/**
 * @brief Fills the volume with 3D points which are to be contained in
 *    any daughters of the input mother volume.
 * @details Requires a proper bounding box from the input volume.
 * @param volume is the input mother volume containing all output points.
 * @param points is the output container, provided by the caller.
 */
template<typename TrackContainer>
VECGEOM_INLINE
void FillContainedPoints(VPlacedVolume const &volume,
                         const double bias,
                         TrackContainer &points,
                         const bool placed = true) {

  const int size = points.capacity();
  points.resize(points.capacity());

  Vector3D<Precision> lower,upper,offset;
  volume.Extent(lower,upper);
  offset = 0.5*(upper+lower);
  const Vector3D<Precision> dim = 0.5*(upper-lower);

  int insideCount = 0;
  std::vector<bool> insideVector(size, false);
  for (int i = 0; i < size; ++i) {
    points.set(i, offset + SamplePoint(dim));
    // measure bias, which is the fraction of points contained in daughters
    for (Vector<Daughter>::const_iterator v = volume.GetDaughters().cbegin(),
         v_end = volume.GetDaughters().cend(); v != v_end; ++v) {
      bool inside = (placed) ? (*v)->Contains(points[i])
                             : (*v)->UnplacedContains(points[i]);
      if (inside) {
        ++insideCount;
        insideVector[i] = true;
      }
    }
  }

  // remove contained points to reduce bias as needed
  int i = 0;
  int tries = 0;
  while (static_cast<double>(insideCount)/static_cast<double>(size) > bias) {
    while (!insideVector[i]) ++i;
    bool contained = false;
    do {
      ++tries;
      if(tries%1000000==0) {
        printf("%s line %i: Warning: %i tries to reduce bias... volume=%s.  Please check.\n", __FILE__, __LINE__, tries, volume.GetLabel().c_str());
      }

      points.set(i, offset + SamplePoint(dim));
      for (Vector<Daughter>::const_iterator v = volume.GetDaughters().cbegin(),
           v_end = volume.GetDaughters().end(); v != v_end; ++v) {
        bool inside = (placed) ? (*v)->Contains(points[i])
                               : (*v)->UnplacedContains(points[i]);
        if (inside) {
          contained = true;
          break;
        }
      }
    } while (contained);
    insideVector[i] = false;
    tries = 0;
    --insideCount;
    ++i;
  }

  // add contained points to increase bias as needed
  i = 0;
  tries = 0;
  while (static_cast<double>(insideCount)/static_cast<double>(size) < bias) {
    while (insideVector[i]) ++i;
    bool contained = false;
    do {
      ++tries;
      if(tries%1000000==0) {
        printf("%s line %i: Warning: %i tries to increase bias... volume=%s.  Please check.\n", __FILE__, __LINE__, tries, volume.GetLabel().c_str());
      }
      const Vector3D<Precision> sample = offset + SamplePoint(dim);
      for (Vector<Daughter>::const_iterator v = volume.GetDaughters().cbegin(),
           v_end = volume.GetDaughters().cend(); v != v_end; ++v) {
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
    tries = 0;
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


/**
 * @brief Fill a container structure (SOA3D or AOS3D) with random
 *    points contained in a volume.
 * @details Input volume must have a valid bounding box, which is used
 *    for sampling.
 * @param volume containing all points
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer>
VECGEOM_INLINE
void FillRandomPoints(VPlacedVolume const &volume,
                      TrackContainer &points) {
  const int size = points.capacity();
  points.resize(points.capacity());

  int tries =0;

  Vector3D<Precision> lower,upper,offset;
  volume.Extent(lower,upper);
  offset = 0.5*(upper+lower);
  const Vector3D<Precision> dim = 0.5*(upper-lower);

  for (int i = 0; i < size; ++i) {
    Vector3D<Precision> point;
    do {
      ++tries;
      if(tries%1000000==0) {
        printf("%s line %i: Warning: %i tries to find contained points... volume=%s.  Please check.\n", __FILE__, __LINE__, tries, volume.GetLabel().c_str());
      }
      point = offset + SamplePoint(dim);
    } while (!volume.UnplacedContains(point));

      //} while (!volume.Contains(point));
    points.set(i, point);
  }
}


/**
 * @brief Fills a container structure (SOA3D or AOS3D) with random
 *    points contained inside a box defined by the two input corners.
 * @param lowercorner, uppercorner define the sampling box
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer>
VECGEOM_INLINE
void FillRandomPoints(Vector3D<Precision> const & lowercorner,
                      Vector3D<Precision> const & uppercorner,
                      TrackContainer &points) {
  const int size = points.capacity();
  points.resize(points.capacity());
  Vector3D<Precision> dim = (uppercorner - lowercorner)/2.;
  Vector3D<Precision> offset = (uppercorner + lowercorner)/2.;
  for (int i = 0; i < size; ++i) {
      points.set(i, offset + SamplePoint(dim));
  }
}

/**
 * @brief Fills a (SOA3D or AOS3D) container with random points inside
 *    a box at the origin
 * @param dim is a Vector3D with w,y,z half-lengths defining the sampling box
 * @param points is the output container, provided by the caller.
 */
template <typename TrackContainer>
VECGEOM_INLINE
void FillRandomPoints(Vector3D<Precision> const & dim,
                      TrackContainer &points) {
  FillRandomPoints( Vector3D<Precision>( -dim.x(), -dim.y(), -dim.z()),
          Vector3D<Precision>(dim.x(),dim.y(),dim.z()), points);
}

/**
 * @brief Generates _uncontained_ global points and directions based
 *   on a logical volume.
 *
 * @details Points and direction coordinates are based on the global
 *   reference frame.  The positions have to be within a given logical
 *   volume, and not within any daughters of that logical volume.
 *
 * The function also returns the generated points in local reference
 *   frame of the logical volume.
 *
 * @param fraction: is the fraction with which the directions should
 *   hit a daughtervolume
 * @param np: number of particles
 *
 */
template <typename TrackContainer>
inline
void FillGlobalPointsAndDirectionsForLogicalVolume(
        LogicalVolume const * lvol,
        TrackContainer  & localpoints,
        TrackContainer  & globalpoints,
        TrackContainer  & directions,
        Precision fraction,
        int np ) {

    // we need to generate a list of all the paths ( or placements ) which reference
    // the logical volume as their deepest node

    std::list<NavigationState *> allpaths;
    GeoManager::Instance().getAllPathForLogicalVolume( lvol, allpaths );

    NavigationState *s1 = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth( ));
    NavigationState *s2 = NavigationState::MakeInstance( GeoManager::Instance().getMaxDepth( ));
    int virtuallyhitsdaughter = 0;
    int reallyhitsdaughter = 0;
    if(allpaths.size() > 0){
        // get one representative of such a logical volume
        VPlacedVolume const * pvol = allpaths.front()->Top();

        // generate points which are in lvol but not in its daughters
        FillUncontainedPoints( *pvol, localpoints );

        // now have the points in the local reference frame of the logical volume
        FillBiasedDirections( *lvol, localpoints, fraction, directions );

        // transform points to global frame
        globalpoints.resize(globalpoints.capacity());
        int placedcount=0;

        while( placedcount < np )
        {
            std::list<NavigationState *>::iterator iter = allpaths.begin();
            while( placedcount < np && iter!=allpaths.end() )
            {
                // this is matrix linking local and global reference frame
                Transformation3D m;
                (*iter)->TopMatrix(m);

                bool hitsdaughter = IsHittingAnyDaughter( localpoints[placedcount], directions[placedcount], *lvol );
                if( hitsdaughter ) virtuallyhitsdaughter++;
                globalpoints.set(placedcount, m.InverseTransform(localpoints[placedcount]));
                directions.set(placedcount, m.InverseTransformDirection(directions[placedcount]));

                // do extensive cross tests
                s1->Clear(); s2->Clear();
                SimpleNavigator nav;
                nav.LocatePoint( GeoManager::Instance().GetWorld( ), globalpoints[placedcount], *s1, true);
                assert( s1->Top()->GetLogicalVolume() == lvol );
                double step = vecgeom::kInfinity;
                nav.FindNextBoundaryAndStep(globalpoints[placedcount], directions[placedcount], *s1, *s2, vecgeom::kInfinity, step );
#ifdef DEBUG
                if( ! hitsdaughter )
                    assert( s1->Distance(*s2) > s2->GetCurrentLevel() - s1->GetCurrentLevel() );
#endif
                if( hitsdaughter )
                    if( s1->Distance(*s2) == s2->GetCurrentLevel() - s1->GetCurrentLevel() ){
                        reallyhitsdaughter++;
                    }

                placedcount++;
                iter++;
            }
        }
    }
    else{
      // an error message
      printf("VolumeUtilities: FillGlobalPointsAndDirectionsForLogicalVolume()... ERROR condition detected.\n");
    }
    printf(" really hits %d, virtually hits %d ", reallyhitsdaughter, virtuallyhitsdaughter );
    NavigationState::ReleaseInstance(s1);
    NavigationState::ReleaseInstance(s2);
}


// same as above; logical volume is given by name
template <typename TrackContainer>
inline
void FillGlobalPointsAndDirectionsForLogicalVolume(
        std::string const & name,
        TrackContainer & localpoints,
        TrackContainer & globalpoints,
        TrackContainer & directions,
        Precision fraction,
        int np ){

    LogicalVolume const * vol = GeoManager::Instance().FindLogicalVolume( name.c_str() );
    if( vol != NULL )
    FillGlobalPointsAndDirectionsForLogicalVolume( vol, localpoints, globalpoints, directions, fraction, np );
}


// same as above; logical volume is given by id
template <typename TrackContainer>
inline
void FillGlobalPointsAndDirectionsForLogicalVolume(
        int id,
        TrackContainer & localpoints,
        TrackContainer & globalpoints,
        TrackContainer & directions,
        Precision fraction,
        int np ){

    LogicalVolume const * vol = GeoManager::Instance().FindLogicalVolume( id );
    if( vol != NULL )
    FillGlobalPointsAndDirectionsForLogicalVolume( vol, localpoints, globalpoints, directions, fraction, np );
}

inline Precision GetRadiusInRing(Precision rmin, Precision rmax) {

  // Generate radius in annular ring according to uniform area
  if (rmin <= 0.) {
    return rmax * std::sqrt( RNG::Instance().uniform() );
  }
  if (rmin != rmax) {
    Precision rmin2 = rmin*rmin;
    Precision rmax2 = rmax*rmax;
    return std::sqrt( rmin2 + RNG::Instance().uniform()*(rmax2 - rmin2) );
  }
  return rmin;
}

} // end namespace volumeUtilities
} } // end global namespace

#endif /* VOLUME_UTILITIES_H_ */
