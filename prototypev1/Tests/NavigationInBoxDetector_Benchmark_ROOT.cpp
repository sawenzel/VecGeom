/*
 * NavigationInBoxDetector_Benchmark_ROOT.cpp
 *
 *  Created on: Feb 3, 2014
 *      Author: swenzel
 */

#include "../TransformationMatrix.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"
#include "../GeoManager.h"
#include "../PhysicalBox.h"
#include "../SimpleVecNavigator.h"
#include <map>
#include <cassert>

#include "TGeoManager.h"
#include "TGeoNavigator.h"

int main(int argc, char * argv[])
{

   // generate benchmark cases
   TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

   double L = 10.;
   double Lz = 10.;
   const double Sqrt2 = sqrt(2.);

   TGeoManager * geom = new TGeoManager("simple1", "ToyDetector");
   TGeoMaterial * matVacuum = new TGeoMaterial("Vacuum",0,0,0);

   TGeoMedium * vac = new TGeoMedium("Vacuum",1,matVacuum);
   TGeoVolume * world = geom->MakeBox("world", vac, L, L, Lz  );

   geom->SetTopVolume( world );

   TGeoVolume * boxlevel2 = geom->MakeBox( "boxlevel2", vac, Sqrt2*L/2./2., Sqrt2*L/2./2.,Lz);
   TGeoVolume * boxlevel3 = geom->MakeBox( "boxlevel3", vac, L/2./2.,L/2./2.,Lz);
   TGeoVolume * boxlevel1 = geom->MakeBox( "boxlevel1", vac, L/2.,L/2., Lz);

   boxlevel2->AddNode( boxlevel3, 1, new TGeoRotation("mat1",0,0,-45));
   boxlevel1->AddNode( boxlevel2, 1, new TGeoRotation("mat2",0,0,45));
   world->AddNode( boxlevel1, 1, new TGeoTranslation("trans1",-L/2., 0,0 ) );
   world->AddNode( boxlevel1, 2, new TGeoTranslation("trans2",L/2., 0,0 ) );
   geom->CloseGeometry();
   delete world->GetVoxels();
   world->SetVoxelFinder(0);

    // perform basic tests
   TGeoNavigator * nav = geom->GetCurrentNavigator(); // new TGeoNavigator(geom);
   StopWatch timer;

   timer.Start();
   int stepsdone=0;
   for(int n=0;n<1000;n++)
     {
   for(int i=0;i<100000;i++)
   // testing the NavigationAndStepInterface
   {
      int localstepsdone=0;
      double distancetravelled=0.;
      Vector3D p;
      PhysicalVolume::samplePoint( p, L, L, Lz, 1. );

      //std::cerr << p << std::endl;
      //setup point in world
      Vector3D d(1,0,0);

      TGeoNode const * vol;
      nav->SetCurrentPoint( p.x, p.y, p.z );
      nav->SetCurrentDirection( d.x, d.y, d.z);
      vol=nav->FindNode(p.x,p.y,p.z);

      while( vol!=NULL )
      {
         localstepsdone++;

         // do one step ( this will internally adjust the current point and so on )
         vol = nav->FindNextBoundaryAndStep(Utils::kInfinity);

         distancetravelled+=nav->GetStep();
         //double const * p = nav->GetCurrentPoint();
         //double const * pl = nav->GetLastPoint();
         //double const * cd = nav->GetCurrentDirection();

         //std::cerr << " proposed step: " << nav->GetStep();
         //std::cerr << " current point " << p[0] << " " << p[1] << " " << p[2] << std::endl;
         //std::cerr << " last point " << pl[0] << " " << pl[1] << " " << pl[2] << std::endl;
         //std::cerr << " current dir " << cd[0] << " " << cd[1] << " " << cd[2] << std::endl;
      }
   //   std::cerr << localstepsdone << " " << distancetravelled << std::endl;
      stepsdone+=localstepsdone;
   }
     }
   timer.Stop();
   std::cout << " time for 100000 particles " << timer.getDeltaSecs( ) << std::endl;
   std::cout << " average steps done " << stepsdone / 100000. << std::endl;
   std::cout << " time per step " << timer.getDeltaSecs()/stepsdone << std::endl;
}


