/*
 * transTest.cpp
 *
 *  Created on: Feb 04, 2014
 *      Author: Guilherme
 */

// PSEUDOCODE:
//   input = [ (0,0,0), (100,0,0), (0,100,0), (0,0,100) ]
//   TransfMatrix = (10,0,0,0,0,0);  // translation along x
//   point2 = MasterToLocal(point,TransfMatrix)
//   print point --> point2          // expected: (10,0,0)

// Can we repeat it using ROOT?
// Can we repeat it using USolids?

//============================

// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
// #include "TGeoMatrix.h"
// #include "TGeoManager.h"
// #include "TGeoVolume.h"
// #include "TGeoShape.h"
// #include "../Utils.h"
#include <iostream>
// #include "mm_malloc.h"
// #include "../GlobalDefs.h"
#include "../GeoManager.h"
//#include "../PhysicalTube.h"
//#include "../PhysicalBox.h"
// #include "../TestShapeContainer.h"
// #include "../SimpleVecNavigator.h"

// in order to compare to USolids
// #include "VUSolid.hh"
// #include "UTubs.hh"

using std::cout;
using std::endl;

int main(int argc, char** argv)
{
  // int np=1024;
  // int NREPS = 1000;
  int np    = 8;
  printf("# points used: NP=%i\n", np);

  Vectors3DSOA input;
  Vectors3DSOA outputx, outputy, outputz, outxyz;
  input.alloc(np);
  outputx.alloc(np);
  outputy.alloc(np);
  outputz.alloc(np);
  outxyz.alloc(np);

// StopWatch timer;

  // generate benchmark cases
  TransformationMatrix const * identity = new TransformationMatrix(0,0,0,0,0,0);

  // the world volume is a box
  double worldsize = 1000.;
  PhysicalVolume * world = GeoManager::MakePlacedBox( new BoxParameters(worldsize, worldsize, worldsize), identity );


  //********  Testing starts here  ***************

  // test points
  input.setFromVector( 0, Vector3D(  0,  0,  0) );
  input.setFromVector( 1, Vector3D(100,  0,  0) );
  input.setFromVector( 2, Vector3D(  0,100,  0) );
  input.setFromVector( 3, Vector3D(  0,  0,100) );
  input.setFromVector( 4, Vector3D(  0,100,100) );
  input.setFromVector( 5, Vector3D(100,  0,100) );
  input.setFromVector( 6, Vector3D(100,100,  0) );
  input.setFromVector( 7, Vector3D(100,100,100) );
  printf("\n*** input:\n");

  for(auto i=0; i<np; ++i) {
    Vector3D temp;
    input.getAsVector(i, temp);
    printf("i=%i - (%f; %f; %f)\n",i,temp.x,temp.y,temp.z);
  }

  // test matrices
  TransformationMatrix testXTransMatrix(10,0,0,0,0,0);
  TransformationMatrix testYTransMatrix(0,10,0,0,0,0);
  TransformationMatrix testZTransMatrix(0,0,10,0,0,0);
  TransformationMatrix testXYZTransMatrix(10,20,30,0,0,0);

  printf("\n*** testX transf matrix:\n");
  testXTransMatrix.print();

  printf("\n*** testY transf matrix:\n");
  testYTransMatrix.print();

  // checks using Vector3D arguments
  testXTransMatrix.MasterToLocal<1,-1,Vc::double_v>(input,outputx);
  testYTransMatrix.MasterToLocal<1,-1,Vc::double_v>(input,outputy);
  testZTransMatrix.MasterToLocal<1,-1,Vc::double_v>(input,outputz);
  testXYZTransMatrix.MasterToLocal<1,-1,Vc::double_v>(input,outxyz);

  printf("\n*** outputx:\n");
  for(auto i=0; i<np; ++i) {
    Vector3D temp;
    //outputx.getAsVector(i, temp);
    temp = outputx.getAsVector(i);
    printf("i=%i - (%f; %f; %f)\n",i,temp.x,temp.y,temp.z);
  }

  printf("\n*** outputy:\n");
  for(auto i=0; i<np; ++i) {
    Vector3D temp;
    outputy.getAsVector(i, temp);
    printf("i=%i - (%f; %f; %f)\n",i,temp.x,temp.y,temp.z);
  }

  printf("\n*** outputz:\n");
  for(auto i=0; i<np; ++i) {
    Vector3D temp;
    outputz.getAsVector(i, temp);
    printf("i=%i - (%f; %f; %f)\n",i,temp.x,temp.y,temp.z);
  }

  printf("\n*** outxyz:\n");
  for(auto i=0; i<np; ++i) {
    Vector3D temp;
    outxyz.getAsVector(i, temp);
    printf("i=%i - (%f; %f; %f)\n",i,temp.x,temp.y,temp.z);
  }

  // cleanup
  input.dealloc();
  outputx.dealloc();
  outputy.dealloc();
  outputz.dealloc();
  outxyz.dealloc();

  return 0;
}
