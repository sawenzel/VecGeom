
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
#include "../PhysicalTube.h"
#include "../TestShapeContainer.h"

// in order to compare to USolids
#include "VUSolid.hh"
#include "UTubs.hh"

using namespace std;

int main(int argc, char *argv[]) {
   int npoints = atoi(argv[1]);

   std::cout << "npoints: " << npoints << std::endl;
	std::cout << "size: " << Vc::double_v::Size << std::endl;
	TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);
	TubeParameters<double> *params = new TubeParameters<double>(10, 20, 30, 0.5, 1.8);

	PlacedUSolidsTube<0, 1296, TubeTraits::HollowTubeWithPhi> *tube = new PlacedUSolidsTube<0, 1296, TubeTraits::HollowTubeWithPhi>(params, identity);

   Vector3D vectors[npoints];
   Vector3DFast vectorsfast[npoints];
   bool contained[npoints];
   bool containedFast[npoints];

   for(int i = 0; i < npoints; i++) {
   	vectors[i].Set( rand() % 10, rand() % 10, rand() % 10 );
   	vectorsfast[i].Set( vectors[i].GetX(), vectors[i].GetY(), vectors[i].GetZ() );
   }

   StopWatch w;

    /* Run contains */ 
    w.Start();
   for(int i = 0; i < npoints; i++) {
   	contained[i] = tube->Contains(vectors[i]);
   }
   w.Stop();
   double t1 = w.getDeltaSecs();

   /* Run contains fast */
   w.Start();
   for(int i = 0; i < npoints; i++) {
      // cout << i << endl;
      // Vector3D tmp(vectorsfast[i].GetX(), vectorsfast[i].GetY(), vectorsfast[i].GetZ() );
      containedFast[i] = tube->Contains(vectorsfast[i]);
      // containedFast[i] = tube->Contains(tmp);
   }
   w.Stop();
   double t2 = w.getDeltaSecs();

   cout << "sizeof(Vector3D): " << sizeof(Vector3D) << endl;
   cout << "sizeof(Vector3DFast): " << sizeof(Vector3DFast) << endl;

   cout << "contains: " << t1 << endl;
   cout << "contains fast: " << t2 << endl;
   cout << "ratio: " << t1 / t2 << endl;


   int wrong = 0, inside = 0;
   for(int i = 0; i < npoints; i++) {
      if(contained[i] != containedFast[i]) {
         wrong++;
         cout << contained[i] << " " << containedFast[i] << endl;
      }
      if(contained[i])
         inside++;
   }
   cout << "wrong: " << wrong << ", inside: " << inside << endl;


   // cout << "Out of " << npoints << " particles, " << hits << " were inside. ( " << ((double)hits/(double)NPOINTS)*100 << "%" << " ) " << endl;


}


