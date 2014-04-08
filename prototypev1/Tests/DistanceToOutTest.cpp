/*
 * File: DistanceToOutTest.cpp
 * Purpose: A simple test for PlacedBox::DistanceToOut() function
 *
 * Change Log:
 *  140314 G.Lima - Created from CHEP13Benchmark
 */

// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoShape.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../PhysicalVolume.h"
#include "../PhysicalTube.h"
#include "../TestShapeContainer.h"
#include "../SimpleVecNavigator.h"

// in order to compare to USolids
//#include "VUSolid.hh"
//#include "UTubs.hh"

#include "Tests/SimpleDetector.hh"

static void cmpresults(double * a1, double * a2, int np,
      PhysicalVolume const * vol, std::vector<Vector3D> const & points, std::vector<Vector3D> const & dirs)
{
   int counter=0;
   for( auto i=0; i<np; ++i )
   {
      if( std::abs( a1[i] - a2[i] ) > Utils::GetCarTolerance() )
      {
         counter++;
#ifdef SHOWDIFFERENCES
         std::cerr << i << " " << a1[i] << " " << a2[i] << std::endl;
         vol->DebugPointAndDirDistanceToIn( points[i], dirs[i] );
#endif
      }
   }
   std::cerr << " have " << counter << " differences " << std::endl;
}


int main(int argc, char** argv)
{
   Vectors3DSOA points, dirs, intermediatepoints, intermediatedirs;
   StructOfCoord rpoints, rintermediatepoints, rdirs, rintermediatedirs;

   int np=3000000;
   int NREPS = 1;
   // int np    = atoi(argv[1]);
   // int NREPS = atoi(argv[2]);
   // std::cout<<"# points used: NP="<< np
   //     <<" / # repetitions: NREPS="<< NREPS << std::endl;

   points.alloc(np);
   dirs.alloc(np);
   intermediatepoints.alloc(np);
   intermediatedirs.alloc(np);

   rpoints.alloc(np);
   rdirs.alloc(np);
   rintermediatepoints.alloc(np);
   rintermediatedirs.alloc(np);

   double *distref = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
   double *distvec = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
   double *distances = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
   double *distances2 = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);
   // double *distancesROOTSCALAR = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
   // double *distancesUSOLIDSCALAR = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
   double *steps = (double *) _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);
   for(auto i=0;i<np;++i) {
     steps[i] = Utils::kInfinity;
     distances[i] = Utils::kInfinity;
     distref[i] = Utils::kInfinity;
     distvec[i] = Utils::kInfinity;
   }

   double *plainpointarray = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);
   double *plaindirtarray = (double *) _mm_malloc(3*np*sizeof(double), ALIGNMENT_BOUNDARY);

   StopWatch timer;

   // construct geometry
   SimpleDetector simpleDetector;
   PhysicalVolume const* world = simpleDetector.getPhysicalVolume();

   //********  Testing starts here  ***************

   world->fillWithRandomPoints(points,np);
   world->fillWithRandomDirections(dirs, np);

   points.toPlainArray(plainpointarray,np);
   dirs.toPlainArray(plaindirtarray,np);

   std::cerr << " Number of daughters " << world->GetNumberOfDaughters() << std::endl;
/*   {
     Vector3D pos,dir;
     for(int ipt=0; ipt<np; ++ipt) {
       points.getAsVector(ipt,pos);
       dirs.getAsVector(ipt,dir);
       printf("Point %i: pos=(%f; %f; %f) dir=(%f; %f; %f)\n", ipt, pos.x, pos.y, pos.z, dir.x, dir.y, dir.z);
       world->PrintDistToEachDaughter(pos,dir);
       world->PrintDistToEachDaughterROOT(pos,dir);
     }
   }
*/

   // time performance for this placement ( we should probably include some random physical steps )

   // do some navigation with a simple Navigator
   // SimpleVecNavigator vecnav(np);
   // const PhysicalVolume ** nextvolumes  = (const PhysicalVolume ** ) _mm_malloc(sizeof(PhysicalVolume *)*np, ALIGNMENT_BOUNDARY);

   {
     PlacedBox<0,1296> const* worldbox = dynamic_cast< PlacedBox<0,1296> const* >(world);

     timer.Start();
     worldbox->DistanceToOutCheck( points, dirs, steps, distref);
     timer.Stop();
     double time1 = timer.getDeltaSecs();

     // calculate distance to Boundary of current volume
     timer.Start();
     worldbox->DistanceToOut( points, dirs, steps, distvec );
     timer.Stop();
     double time2 = timer.getDeltaSecs();

     printf("Timing: %f %f\n",time1,time2);
     for(auto i=0; i<np; ++i) if(abs(distref[i]-distvec[i])>1.e-10) printf("DistToOutCheck: i=%i distref=%f distvec=%f\n", i, distref[i], distvec[i]);
        }

/*
   timer.Start();
   for(int reps=0 ;reps < NREPS; reps++ )
   {
      vecnav.DistToNextBoundary( world, points, dirs, steps, distances, nextvolumes , np );
   }
   timer.Stop();
   double t0 = timer.getDeltaSecs();
   std::cerr << t0 <<" <-- Time for vecnav.DistToNextBoundary()"<< std::endl;
   // give out hit pointers
   double d0=0.;
   for(auto k=0;k<np;k++)
   {
      d0+=distances[k];
      distances2[3*k] = distances[k];
      distances[k]=Utils::kInfinity;
   }


   timer.Start();
   for(int reps=0 ;reps < NREPS; reps++ )
   {
      vecnav.DistToNextBoundaryUsingUnplacedVolumes( world, points, dirs, steps, distances, nextvolumes , np );
   }
   timer.Stop();
   double t1= timer.getDeltaSecs();

   std::cerr << t1 <<" <-- Time for vecnac.DistToNextBoundaryUsingUnplacedVolumes"<< std::endl;

   double d1=0.;
   for(auto k=0;k<np;k++)
   {
      d1+=distances[k];
      distances2[3*k+1] = distances[k];
      distances[k]=Utils::kInfinity;
   }

   // now using the ROOT Geometry library (scalar version)
   timer.Start();
   for(int reps=0;reps < NREPS; reps ++ )
   {
      vecnav.DistToNextBoundaryUsingROOT( world, plainpointarray, plaindirtarray, steps, distances, nextvolumes, np );
   }
   timer.Stop();
   double t3 = timer.getDeltaSecs();
   std::cerr << t3 <<" <-- Time for vecnav.DistToNextBoundaryUsingROOT"<< std::endl;

   std::cerr <<"Ratio of times Unpl/Vect: "<< t1/t0 << std::endl;
   std::cerr <<"Ratio of times ROOT/Vect: "<< t3/t0 << std::endl;

   double d3=0;
   for(auto k=0;k<np;k++)
   {
        d3+=distances[k];
       distances2[3*k+2] = distances[k];
   }
   std::cerr <<"Comparing results: "<< d0 << " " << d1 << " " << d3 << std::endl;

   bool first=true;
   for(auto k=0;k<np;++k) {
     if( fabs(distances2[3*k+1]-distances2[3*k]) > 0.1 ) {
       if(first) {
         std::cerr<<"Comparing individual measurements: Point#  distances(Vec,VecUnplVol,ROOT)...\n";
         first = false;
       }
       std::cerr <<"Point# "<< k <<':'
            <<' '<< distances2[3*k]
            <<' '<< distances2[3*k+1]
            <<' '<< distances2[3*k+2]
            << std::endl;
     }
   }
*/
   //vecnav.DistToNextBoundaryUsingUnplacedVolumes( world, points, dirs, steps, distances, nextvolumes , np );
   //( world, points, dirs,  );


   // give out hit pointers
   /*
   for(auto k=0;k<np;k++)
   {
      if( nextvolumes[k] !=0 )
      {
         nextvolumes[k]->printInfo();
      }
      else
      {
         std::cerr << "hitting boundary of world"  << std::endl;
      }
   }
*/
    _mm_free(distances);
    return 1;
}
