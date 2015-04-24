/*
 *  ABBoxBenchmark.cpp
 *
 *  simple benchmark to play with aligned bounding boxes
 *  which could be a simple but efficient form of "voxelization"
 *
 *  bounding boxes are setup in some form of regular 3D grid of size N x N x N
 *
 *  benchmark:
 *  a) intersecting bounding boxes without caching of inverseray + no vectorization
 *  b) intersecting bounding boxes with caching of inverseray + no vectorization
 *  c) intersecting bounding boxes with caching + vectorization
 *
 *  Created on: 20.04.2015
 *      Author: swenzel
 */

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "benchmarking/Benchmarker.h"
#include "volumes/kernel/BoxImplementation.h"
#include "volumes/utilities/VolumeUtilities.h"
#include "management/GeoManager.h"
#include "ArgParser.h"
#include "base/SOA3D.h"
#include "base/Stopwatch.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <list>
#include <utility>
#include <backend/vc/Backend.h>
#include <backend/Backend.h>

using namespace vecgeom;

// #define INNERTIMER
#define SORTHITBOXES

typedef std::pair<int, double> BoxIdDistancePair_t;
//typedef std::list<BoxIdDistancePair_t> Container_t;
typedef std::vector<BoxIdDistancePair_t> Container_t;

// build an abstraction of sort to sort vectors and lists portably
template<typename C, typename Compare> void sort(C & v, Compare cmp) {
    sort(v.begin(), v.end(), cmp);
}


// comparator for hit boxes: returns true if left is < right
__attribute__((always_inline))
bool HitBoxComparator( BoxIdDistancePair_t const & left, BoxIdDistancePair_t const & right ){
    return left.second < right.second;
}


struct HitBoxComparatorFunctor{
  bool operator()( BoxIdDistancePair_t const & left, BoxIdDistancePair_t const & right )
  {
    return left.second < right.second;
  }
};

// using FP_t = bool (*)( BoxIdDistancePair_t const & left, BoxIdDistancePair_t const & right );
using FP_t = HitBoxComparatorFunctor;

// template specialization for list
template<>
void sort<std::list<BoxIdDistancePair_t>, FP_t >( std::list<BoxIdDistancePair_t> & v, FP_t cmp){
    v.sort(cmp);
}


// output for hitboxes
template <typename stream>
stream & operator<<(stream & s, std::list<BoxIdDistancePair_t> const & list){
    for(auto i : list){
        s << "(" << i.first << "," << i.second << ")" << " ";
    }
    return s;
}

__attribute__((noinline))
int benchNoCachingNoVector( Vector3D<Precision> const & point,
                            Vector3D<Precision> const & dir,
                            std::vector<Vector3D<Precision> > const &
#ifdef SORTHITBOXES
                            , Container_t & hitlist
#endif
                          );

__attribute__((noinline))
int benchCachingNoVector(   Vector3D<Precision> const & point,
                            Vector3D<Precision> const & dir,
                            std::vector<Vector3D<Precision> > const &
#ifdef SORTHITBOXES
                            , Container_t  & hitlist
#endif
);

__attribute__((noinline))
int benchCachingAndVector(  Vector3D<Precision> const & point,
                            Vector3D<Precision> const & dir,
                            Vector3D<kVc::precision_v> const *, int number
#ifdef SORTHITBOXES
                            , Container_t  & hitlist
#endif
);


#define N 20        // boxes per dimension
#define SZ 10000      // samplesize
double delta = 0.5; // if delta > 1 the boxes will overlap


int main(){
    // number of boxes
    int numberofboxes = N*N*N;

    int code = (2 << 1) + (2 << 0) + (2 << 1);
    std::cerr << code << "\n";

    // setup AOS form of boxes
    std::vector<Vector3D<Precision> > uppercorners(numberofboxes);
    std::vector<Vector3D<Precision> > lowercorners(numberofboxes);

    // setup same in mixed array of corners ... upper-lower-upper-lower ...
    std::vector<Vector3D<Precision> > corners(2*numberofboxes);

    // setup SOA form of boxes -- the memory layout should probably rather be SOA6D
    Vector3D<kVc::precision_v> * VcCorners =  new Vector3D<kVc::precision_v>[ 2*numberofboxes/kVc::precision_v::Size ];

    int counter1=0;
    int counter2=0;
    for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            for(int k=0; k<N; ++k){
                lowercorners[counter1] = Vector3D<Precision>( i, j, k );
                uppercorners[counter1] = Vector3D<Precision>( i + delta, j + delta, k + delta );

                corners[counter2] = lowercorners[counter1];
                counter2++;
                corners[counter2] = uppercorners[counter1];
                counter2++;

                counter1++;
            }
        }
    }

    // print boxes
    for(int i=0; i<numberofboxes; ++i)
    {
        // std::cerr << "# " << i << " lower " << lowercorners[i] << " " << uppercorners[i] << "\n";
    }

    // set up VcCorners
    counter2=0;
    for(int i=0; i < numberofboxes; i+=kVc::precision_v::Size)
    {
        Vector3D<kVc::precision_v> lower;
        Vector3D<kVc::precision_v> upper;
        // assign by components
        for( int k=0;k<kVc::precision_v::Size;++k ){
            lower.x()[k] = lowercorners[i+k].x();
            lower.y()[k] = lowercorners[i+k].y();
            lower.z()[k] = lowercorners[i+k].z();
            upper.x()[k] = uppercorners[i+k].x();
            upper.y()[k] = uppercorners[i+k].y();
            upper.z()[k] = uppercorners[i+k].z();
        }
        //std::cerr << lower << "\n";
        //std::cerr << upper << "\n";
        VcCorners[counter2++] = lower;
        VcCorners[counter2++] = upper;
    }
    std::cerr << "assigned " << counter2 << "Vc vectors\n";

    // constructing samples
    std::vector<Vector3D<Precision> > points(SZ);
    std::vector<Vector3D<Precision> > directions(SZ);
    for(int i=0;i<SZ;++i)
    {
        points[i] = Vector3D<Precision>( N*delta + 0.1, N*delta + 0.1, N*delta + 0.1 );
        directions[i] = volumeUtilities::SampleDirection();
    }

    Container_t hitlist;
    hitlist.resize(2*N);

    Stopwatch timer;
    int hits=0;
    double meanfurthestdistance = 0;

    timer.Start();
    for(int i=0;i<SZ;++i){
#ifdef SORTHITBOXES
        hitlist.clear();
#endif
        hits+=benchCachingNoVector( points[i], directions[i], corners
#ifdef SORTHITBOXES
        , hitlist
#endif
        );
#ifdef SORTHITBOXES
       sort( hitlist, HitBoxComparatorFunctor() );
       meanfurthestdistance+=hitlist.back().second;
       // std::cerr << hitlist << "\n";
#endif
    }
    timer.Stop();
    std::cerr << "Cached times and hit " << timer.Elapsed() << " " << hits << " " << meanfurthestdistance << "\n";

    hits=0;
    meanfurthestdistance=0.;
    timer.Start();
    for(int i=0;i<SZ;++i){
        hitlist.clear();
        hits+=benchNoCachingNoVector( points[i], directions[i], corners
#ifdef SORTHITBOXES
        , hitlist
#endif
        );
#ifdef SORTHITBOXES
        sort(hitlist, HitBoxComparatorFunctor() );
        meanfurthestdistance+=hitlist.back().second;
#endif
    }
    timer.Stop();
    std::cerr << "Ordinary times and hit " << timer.Elapsed() << " " << hits << " " << meanfurthestdistance << "\n";

    hits=0;
    meanfurthestdistance=0.;
    timer.Start();
    for(int i=0;i<SZ;++i){
#ifdef SORTHITBOXES
        hitlist.clear();
#endif
        hits+=benchCachingAndVector( points[i], directions[i], VcCorners, numberofboxes/kVc::precision_v::Size
#ifdef SORTHITBOXES
        , hitlist
#endif
        );
#ifdef SORTHITBOXES
       sort( hitlist,  HitBoxComparatorFunctor() );
       meanfurthestdistance+=hitlist.back().second;
       //std::cerr << "VECTORHITLIST" << hitlist << "\n";
#endif
    }
    timer.Stop();
    std::cerr << "Vector times and hit " << timer.Elapsed() << " " << hits << " " << meanfurthestdistance << "\n";

    return 0;
}


int benchNoCachingNoVector( Vector3D<Precision> const & point,
                            Vector3D<Precision> const & dir,
                            std::vector<Vector3D<Precision> > const & corners
#ifdef SORTHITBOXES
                            , Container_t & hitlist
#endif
                          ){
#ifdef INNERTIMER
    Stopwatch timer;
    timer.Start();
#endif
    int vecsize = corners.size() / 2;
    int hitcount = 0;
    for( auto box = 0; box < vecsize; ++box ){
         double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::Intersect(
                &corners[2*box],
                point,
                dir,
                0, vecgeom::kInfinity );
         if( distance < vecgeom::kInfinity ){
             hitcount++;
#ifdef SORTHITBOXES
             hitlist.push_back( BoxIdDistancePair_t( box, distance) );
#endif
         }
    }
#ifdef INNERTIMER
    timer.Stop();
    std::cerr << "# ORDINARY hitting " << hitcount << "\n";
    std::cerr << "# ORDINARY timer " << timer.Elapsed() << "\n";
#endif
    return hitcount;
}

int benchCachingNoVector( Vector3D<Precision> const & point,
                          Vector3D<Precision> const & dir,
                          std::vector<Vector3D<Precision> > const & corners
#ifdef SORTHITBOXES
                            , Container_t  & hitlist
#endif
){
#ifdef INNERTIMER
    Stopwatch timer;
    timer.Start();
#endif

    Vector3D<Precision> invdir(1./dir.x(), 1./dir.y(), 1./dir.z());
    int vecsize = corners.size() / 2;
    int hitcount = 0;
    int sign[3]; sign[0] = invdir.x() < 0; sign[1] = invdir.y() < 0; sign[2] = invdir.z() < 0;
    // interpret as binary number and do a switch statement
    // do a big switch statement here
   // int code = 2 << size[0] + 2 << size[1] + 2 << size[2];
    for( auto box = 0; box < vecsize; ++box ){
         double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel2<kScalar>(
            &corners[2*box],
            point,
           invdir,
           sign[0],sign[1],sign[2],
            0, vecgeom::kInfinity );
            if( distance < vecgeom::kInfinity ){
                hitcount++;
#ifdef SORTHITBOXES
             hitlist.push_back( BoxIdDistancePair_t( box, distance) );
#endif
            }
        }

    //    switch( size[0] + size[1] + size[2] ){
//    case 0: {
//        for( auto box = 0; box < vecsize; ++box ){
//        double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel<kScalar,0,0,0>(
//           &corners[2*box],
//           point,
//           invdir,
//           0, vecgeom::kInfinity );
//           if( distance < vecgeom::kInfinity ) hitcount++;
//         }       break; }
//    case 3: {
//        for( auto box = 0; box < vecsize; ++box ){
//                double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel<kScalar,1,1,1>(
//                   &corners[2*box],
//                   point,
//                   invdir,
//                   0, vecgeom::kInfinity );
//                   if( distance < vecgeom::kInfinity ) hitcount++;
//                 }       break; }
//    default : std::cerr << "DEFAULT CALLED\n";
//    }
#ifdef INNERTIMER
    timer.Stop();
    std::cerr << "# CACHED hitting " << hitcount << "\n";
    std::cerr << "# CACHED timer " << timer.Elapsed() << "\n";
#endif
    return hitcount;
}


int benchCachingAndVector( Vector3D<Precision> const & point, Vector3D<Precision> const & dir,
        Vector3D<kVc::precision_v> const * corners, int vecsize
#ifdef SORTHITBOXES
                            , Container_t & hitlist
#endif
){
#ifdef INNERTIMER
    Stopwatch timer;
    timer.Start();
#endif

    Vector3D<Precision> invdir(1./dir.x(), 1./dir.y(), 1./dir.z());
    int hitcount = 0;


    int sign[3]; sign[0] = invdir.x() < 0; sign[1] = invdir.y() < 0; sign[2] = invdir.z() < 0;
    for( auto box = 0; box < vecsize; ++box ){
            kVc::precision_v distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel2<kVc>(
               &corners[2*box],
               point,
               invdir,sign[0],sign[1],sign[2],
               0, vecgeom::kInfinity );
               kVc::bool_v hit = distance < vecgeom::kInfinity;
               //std::cerr << hit << "\n";
               // this is Vc specific
               hitcount += hit.count();
               #ifdef SORTHITBOXES
// a little tricky: need to iterate over the mask
               for(auto i=0; i < kVc::precision_v::Size; ++i){
                   if( hit[i] )
                       // which box id??
                   hitlist.push_back( BoxIdDistancePair_t( box * kVc::precision_v::Size + i, distance[i]) );
               }
               #endif
    }
    // interpret as binary number and do a switch statement
    // do a big switch statement here
//    switch( size[0] + size[1] + size[2] ){
//    case 0: {
//        for( auto box = 0; box < vecsize; ++box ){
//        kVc::precision_v distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel<kVc,0,0,0>(
//           &corners[2*box],
//           point,
//           invdir,
//           0, vecgeom::kInfinity );
//           kVc::bool_v hit = distance < vecgeom::kInfinity;
//           //std::cerr << hit << "\n";
//           hitcount += hit.count();
//        }       break; }
//    case 3: {
//        for( auto box = 0; box < vecsize; ++box ){
//          kVc::precision_v distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel<kVc,1,1,1>(
//          &corners[2*box],
//          point,
//          invdir,
//          0, vecgeom::kInfinity );
//          kVc::bool_v hit = distance < vecgeom::kInfinity;
//          //std::cerr << hit << "\n";
//          hitcount += hit.count();
//    }       break; }
//    default : std::cerr << "DEFAULT CALLED\n";
//    }
#ifdef INNERTIMER
    timer.Stop();
    std::cerr << "# VECTOR hitting " << hitcount << "\n";
    std::cerr << "# VECTOR timer " << timer.Elapsed() << "\n";
#endif
    return hitcount;
}

