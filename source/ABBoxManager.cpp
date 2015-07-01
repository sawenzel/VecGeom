/*
 * ABBoxManager.cpp
 *
 *  Created on: 24.04.2015
 *      Author: swenzel
 */

#include "navigation/ABBoxNavigator.h"
#include "volumes/UnplacedBox.h"

#ifdef VECGEOM_VC
//#include "backend/vc/Backend.h"
#include "backend/vcfloat/Backend.h"
#else
#include "backend/scalarfloat/Backend.h"
#endif

#include <cassert>

namespace vecgeom {
inline namespace cxx {

 int ABBoxNavigator::GetHitCandidates(
                         LogicalVolume const * lvol,
                         Vector3D<Precision> const & point,
                         Vector3D<Precision> const & dir,
                         ABBoxManager::ABBoxContainer_t const & corners, int size,
                         ABBoxManager::HitContainer_t & hitlist) const {

    Vector3D<Precision> invdir(1./dir.x(), 1./dir.y(), 1./dir.z());
    int vecsize = size;
    int hitcount = 0;
    int sign[3]; sign[0] = invdir.x() < 0; sign[1] = invdir.y() < 0; sign[2] = invdir.z() < 0;
    // interpret as binary number and do a switch statement
    // do a big switch statement here
   // int code = 2 << size[0] + 2 << size[1] + 2 << size[2];
    for( auto box = 0; box < vecsize; ++box ){
         double distance = BoxImplementation<translation::kIdentity, rotation::kIdentity>::IntersectCachedKernel2<kScalar, double>(
            &corners[2*box],
            point,
           invdir,
           sign[0],sign[1],sign[2],
            0, vecgeom::kInfinity );
            if( distance < vecgeom::kInfinity ){
                hitcount++;
             hitlist.push_back( ABBoxManager::BoxIdDistancePair_t( box, distance) );
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

 // vector version
 int ABBoxNavigator::GetHitCandidates_v(
                          LogicalVolume const * lvol,
                          Vector3D<Precision> const & point,
                          Vector3D<Precision> const & dir,
                          ABBoxManager::ABBoxContainer_v const & corners, int size,
                          ABBoxManager::HitContainer_t & hitlist) const {

#ifdef VECGEOM_VC
     Vector3D<float> invdirfloat(1.f/(float)dir.x(), 1.f/(float)dir.y(), 1.f/(float)dir.z());
     Vector3D<float> pfloat((float)point.x(), (float)point.y(), (float)point.z());

     int vecsize = size;
     int hitcount = 0;
     int sign[3]; sign[0] = invdirfloat.x() < 0; sign[1] = invdirfloat.y() < 0; sign[2] = invdirfloat.z() < 0;
     for( auto box = 0; box < vecsize; ++box ){
          ABBoxManager::Real_v distance = BoxImplementation<translation::kIdentity,
                  rotation::kIdentity>::IntersectCachedKernel2<kVcFloat, ABBoxManager::Real_t>(
                        &corners[2*box], pfloat, invdirfloat, sign[0], sign[1], sign[2], 0,
                        static_cast<float>(vecgeom::kInfinity) );
          ABBoxManager::Bool_v hit = distance < static_cast<float>(vecgeom::kInfinity);
          // this is Vc specific
          // a little tricky: need to iterate over the mask -- this does not easily work with scalar types
          for(auto i=0; i < kVcFloat::precision_v::Size; ++i){
          if( hit[i] )
              hitlist.push_back( ABBoxManager::BoxIdDistancePair_t( box * kVcFloat::precision_v::Size + i, distance[i]) );
          }
     }
     return hitcount;
#else
     Vector3D<float> invdirfloat(1.f/(float)dir.x(), 1.f/(float)dir.y(), 1.f/(float)dir.z());
     Vector3D<float> pfloat((float)point.x(), (float)point.y(), (float)point.z());

     int vecsize = size;
     int hitcount = 0;
     int sign[3]; sign[0] = invdirfloat.x() < 0; sign[1] = invdirfloat.y() < 0; sign[2] = invdirfloat.z() < 0;
     for( auto box = 0; box < vecsize; ++box ){
          float distance = BoxImplementation<translation::kIdentity,
                  rotation::kIdentity>::IntersectCachedKernel2<kScalarFloat, float >(
                        &corners[2*box], pfloat, invdirfloat, sign[0], sign[1], sign[2], 0,
                        static_cast<float>(vecgeom::kInfinity) );
          bool hit = distance < static_cast<float>(vecgeom::kInfinity);
          if (hit) hitlist.push_back( ABBoxManager::BoxIdDistancePair_t( box , distance ) );
     }
     return hitcount;
#endif

 }

void ABBoxManager::ComputeABBox( VPlacedVolume const * pvol, ABBox_t * lowerc, ABBox_t * upperc ) {
        // idea: take the 8 corners of the bounding box in the reference frame of pvol
        // transform those corners and keep track of minimum and maximum extent
        // TODO: could make this code shorter with a more complex Vector3D class
        Vector3D<Precision> lower, upper;
        pvol->Extent( lower, upper );
        Vector3D<Precision> delta = upper-lower;
        Precision minx,miny,minz,maxx,maxy,maxz;
        minx = kInfinity;
        miny = kInfinity;
        minz = kInfinity;
        maxx = -kInfinity;
        maxy = -kInfinity;
        maxz = -kInfinity;
        Transformation3D const * transf = pvol->GetTransformation();
        for(int x=0;x<=1;++x)
            for(int y=0;y<=1;++y)
                for(int z=0;z<=1;++z){
                      Vector3D<Precision> corner;
                      corner.x() = lower.x() + x*delta.x();
                      corner.y() = lower.y() + y*delta.y();
                      corner.z() = lower.z() + z*delta.z();
                      Vector3D<Precision> transformedcorner =
                        transf->InverseTransform( corner );
                      minx = std::min(minx, transformedcorner.x());
                      miny = std::min(miny, transformedcorner.y());
                      minz = std::min(minz, transformedcorner.z());
                      maxx = std::max(maxx, transformedcorner.x());
                      maxy = std::max(maxy, transformedcorner.y());
                      maxz = std::max(maxz, transformedcorner.z());
                }
        *lowerc = Vector3D<Precision>(minx - 1E-3 ,miny - 1E-3, minz - 1E-3);
        *upperc = Vector3D<Precision>(maxx + 1E-3 ,maxy +  1E-3,maxz + 1E-3 );

#ifdef CHECK
        // do some tests on this stuff
        delta = (*upperc - *lowerc)/2.;
        Vector3D<Precision> boxtranslation = (*lowerc + *upperc)/2.;
        UnplacedBox box(delta);
        Transformation3D tr( boxtranslation.x(), boxtranslation.y(), boxtranslation.z() );
        VPlacedVolume const * boxplaced = LogicalVolume("",&box).Place(&tr);
        // no point on the surface of the aligned box should be inside the volume
        std::cerr << "lower " << *lowerc;
        std::cerr << "upper " << *upperc;
        int contains = 0;
        for(int i=0;i<10000;++i)
        {
            Vector3D<Precision> p =  box.GetPointOnSurface() + boxtranslation;
            std::cerr << *lowerc << " " << * upperc << " " << p << "\n";
            if( pvol->Contains( p ) ) contains++;
        }
        if( contains > 10){
            Visualizer visualizer;
            visualizer.AddVolume(*pvol, *pvol->GetTransformation());
            visualizer.AddVolume(*boxplaced, tr );
            visualizer.Show();
        }
        std::cerr << "## wrong points " << contains << "\n";
#endif
}

void ABBoxManager::InitABBoxes( LogicalVolume const * lvol ){
        if( fVolToABBoxesMap.find(lvol) != fVolToABBoxesMap.end() )
        {
            // remove old boxes first
            RemoveABBoxes(lvol);
        }
        uint ndaughters = lvol->GetDaughtersp()->size();
        ABBox_t * boxes = new ABBox_t[ 2*ndaughters ];
        fVolToABBoxesMap[lvol] = boxes;

        // same for the vector part
        int extra = (ndaughters % Real_vSize > 0) ? 1 : 0;
        int size = 2 * ( ndaughters / Real_vSize + extra );
        ABBox_v * vectorboxes =  new ABBox_v[ size ];
        fVolToABBoxesMap_v[lvol] = vectorboxes;

        // calculate boxes by iterating over daughters
        for(uint d=0;d<ndaughters;++d){
            auto pvol = lvol->GetDaughtersp()->operator [](d);
            ComputeABBox( pvol, &boxes[2*d], &boxes[2*d+1] );
#ifdef CHECK
            // do some tests on this stuff
            Vector3D<Precision> lower = boxes[2*d];
            Vector3D<Precision> upper = boxes[2*d+1];

            Vector3D<Precision> delta = (upper - lower)/2.;
            Vector3D<Precision> boxtranslation = (lower + upper)/2.;
            UnplacedBox box(delta);
            Transformation3D tr( boxtranslation.x(), boxtranslation.y(), boxtranslation.z() );
            VPlacedVolume const * boxplaced = LogicalVolume("",&box).Place(&tr);
//                   int contains = 0;
//                   for(int i=0;i<10000;++i)
//                   {
//                       Vector3D<Precision> p =  box.GetPointOnSurface() + boxtranslation;
//                       std::cerr << *lowerc << " " << * upperc << " " << p << "\n";
//                       if( pvol->Contains( p ) ) contains++;
//                   }
//                   if( contains > 10){
#endif
        }


        // initialize vector version of Container
        int index=0;
        unsigned int assignedscalarvectors=0;
        for( uint i=0; i < ndaughters; i+= Real_vSize )
        {
            Vector3D<Real_v> lower;
            Vector3D<Real_v> upper;
            // assign by components ( this will be much more generic with new VecCore )
#ifdef VECGEOM_VC
            for( uint k=0;k<Real_vSize;++k ){
                if(2*(i+k) < 2*ndaughters )
                {
                    lower.x()[k] = boxes[2*(i+k)].x();
                    lower.y()[k] = boxes[2*(i+k)].y();
                    lower.z()[k] = boxes[2*(i+k)].z();
                    upper.x()[k] = boxes[2*(i+k)+1].x();
                    upper.y()[k] = boxes[2*(i+k)+1].y();
                    upper.z()[k] = boxes[2*(i+k)+1].z();
                    assignedscalarvectors+=2;
                }
                else{
                    // filling in bounding boxes of zero size
                    // better to put some irrational number than 0?
                    lower.x()[k] = 0.;
                    lower.y()[k] = 0.;
                    lower.z()[k] = 0.;
                    upper.x()[k] = 0.;
                    upper.y()[k] = 0.;
                    upper.z()[k] = 0.;
                }
             }
            vectorboxes[index++] = lower;
            vectorboxes[index++] = upper;
        }
#else
        lower.x() = boxes[2*i].x();
        lower.y() = boxes[2*i].y();
        lower.z() = boxes[2*i].z();
        upper.x() = boxes[2*i+1].x();
        upper.y() = boxes[2*i+1].y();
        upper.z() = boxes[2*i+1].z();
        assignedscalarvectors+=2;

        vectorboxes[index++] = lower;
        vectorboxes[index++] = upper;
    }
#endif
        assert( index == size );
        assert( assignedscalarvectors == 2*ndaughters );
}



void ABBoxManager::RemoveABBoxes( LogicalVolume const * lvol){
        if( fVolToABBoxesMap.find(lvol) != fVolToABBoxesMap.end() ) {
            delete[] fVolToABBoxesMap[lvol];
            fVolToABBoxesMap.erase(lvol);
        }
    }
}




}
