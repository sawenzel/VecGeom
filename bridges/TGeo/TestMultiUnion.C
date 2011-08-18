#include "UBox.hh"
#include "UMultiUnion.hh"
#include "UTransform3D.hh"
#include "UVoxelFinder.hh"
#include "TGeoUShape.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TStopwatch.h"
#include "TRandom.h"
#include "TPolyMarker3D.h"
/*
void TestMultiUnion()
{
   // Initialization of ROOT environment:
   // Test for a multiple union solid.
   TGeoManager *geom = new TGeoManager("UMultiUnion","Test of a UMultiUnion");
   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
   TGeoMedium *Al = new TGeoMedium("Root Material",2, matAl);
   
   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 2000., 2000., 2000.);
   TGeoShape *tgeobox = top->GetShape();
   geom->SetTopVolume(top);

   // Instance:
      // Creation of several nodes:
   UBox *box = new UBox("UBox",40,40,40);

   TGeoUShape *Shape1 = new TGeoUShape("Shape1",box);
   TGeoVolume *Volume1 = new TGeoVolume("Volume1",Shape1);
   Volume1->SetLineColor(1);
   
      // Number of nodes to implement:
   int numNodesImpl = 500;
   int mIndex = 0, nIndex = 0, oIndex = 0;
	int carBoxesX = 20;
	int carBoxesY = 20;
	int carBoxesZ = 20;   
   UTransform3D* arrayTransformations[numNodesImpl];
	TGeoCombiTrans* arrayCombiTrans[numNodesImpl];
  
     // Constructor:
   UMultiUnion *multi_union = new UMultiUnion("multi_union");     
   
      // Transformation:
   for(mIndex = 0 ; (mIndex < numNodesImpl) && (mIndex < carBoxesX*carBoxesY*carBoxesZ) ; mIndex++)
   {
      arrayTransformations[mIndex] = new UTransform3D(-2000+50+2*50*(mIndex%carBoxesX),-2000+50+2*50*nIndex,-2000+50+2*50*oIndex,0,0,0);
      multi_union->AddNode(box,arrayTransformations[mIndex]);
      
      arrayCombiTrans[mIndex] = new TGeoCombiTrans(-2000+50+2*50*(mIndex%carBoxesX),-2000+50+2*50*nIndex,-2000+50+2*50*oIndex,new TGeoRotation("rot",0,0,0));      
      top->AddNode(Volume1,mIndex+1,arrayCombiTrans[mIndex]);      
      
      // Preparing "Draw":
      if((nIndex%carBoxesY)==(carBoxesY-1) && (mIndex%carBoxesX)==(carBoxesX-1))         
      {
         nIndex = -1;
         oIndex++;
      }      
      
		if((mIndex%carBoxesX)==(carBoxesX-1))
		{
			nIndex++;
		}
   }                                                                                                                                   

   geom->CloseGeometry();

   // Voxelize "multi_union"
   multi_union -> Voxelize();

   cout << "[> DisplayVoxelLimits:" << endl;   
   multi_union -> fVoxels -> DisplayVoxelLimits();   

   cout << "[> DisplayBoundaries:" << endl;      
   multi_union -> fVoxels -> DisplayBoundaries();   

   cout << "[> BuildListNodes:" << endl;      
   multi_union -> fVoxels -> DisplayListNodes();

   cout << "[> Test:" << endl;   
   
   double bmin[3], bmax[3];
   UVector3 point;
   multi_union->Extent(bmin, bmax);
   int npoints = 1000000;
   TPolyMarker3D *pminside = new TPolyMarker3D();
   pminside->SetMarkerColor(kRed);
   TPolyMarker3D *pmoutside = new TPolyMarker3D();
   pmoutside->SetMarkerColor(kGreen);
   
   TStopwatch timer;
   double chrono = 0.;
      
   int n10 = npoints/10;
   
   for (int ipoint = 0; ipoint<npoints; ipoint++)
   {
      if (n10 && (ipoint%n10)==0) printf("test inside ... %d%%\n",int(100*ipoint/npoints));
      point.x = gRandom->Uniform(bmin[0], bmax[0]);
      point.y = gRandom->Uniform(bmin[1], bmax[1]);
      point.z = gRandom->Uniform(bmin[2], bmax[2]);

      timer.Start();      
      VUSolid::EnumInside inside = multi_union->Inside(point);
      timer.Stop();
      chrono += timer.CpuTime();
      timer.Reset();            

      if (inside==VUSolid::eInside || inside==VUSolid::eSurface)
      {
         pminside->SetNextPoint(point.x, point.y, point.z);
      }
      else
      {
         pmoutside->SetNextPoint(point.x, point.y, point.z);
      }   
   }

//   timer.Print();
   cout << "CPUTIME: " << chrono << endl;
   geom->GetTopVolume()->Draw();
   pminside->Draw();
//   pmoutside->Draw();

   // Program comes to an end:
   printf("[> END\n");
}
*/

// TO TEST THE METHODS SAFETY AND NORMAL
void TestMultiUnion()
{
   // Initialization of ROOT environment:
   // Test for a multiple union solid.
   TGeoManager *geom = new TGeoManager("UMultiUnion","Test of a UMultiUnion");
   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
   TGeoMedium *Al = new TGeoMedium("Root Material",2, matAl);
   
   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 1000., 1000., 1000.);
   TGeoShape *tgeobox = top->GetShape();
   geom->SetTopVolume(top);

   // Instance:
      // Creation of several nodes:
   UBox *box = new UBox("UBox",200,200,200);
  
     // Constructor:
   UMultiUnion *multi_union = new UMultiUnion("multi_union");     
   UTransform3D* trans = new UTransform3D(0,0,0,0,0,0);
   UTransform3D* trans2 = new UTransform3D(200,0,0,0,0,0);      
   UTransform3D* trans3 = new UTransform3D(50,400,0,0,0,45);       
   
   multi_union->AddNode(box,trans);
   multi_union->AddNode(box,trans2); 
   multi_union->AddNode(box,trans3);       
                                                                                                                                         
   geom->CloseGeometry();
   
   TGeoUShape *Shape1 = new TGeoUShape("Shape1",box);
   TGeoVolume *Volume1 = new TGeoVolume("Volume1",Shape1);
   Volume1->SetLineColor(2);   
      
   TGeoCombiTrans* transf1 = new TGeoCombiTrans(0,0,0,new TGeoRotation("rot1",0,0,0));      
   top->AddNode(Volume1,1,transf1);

   TGeoCombiTrans* transf2 = new TGeoCombiTrans(200,0,0,new TGeoRotation("rot1",0,0,0));      
   top->AddNode(Volume1,1,transf2);

   TGeoCombiTrans* transf3 = new TGeoCombiTrans(50,400,0,new TGeoRotation("rot1",0,0,45));      
   top->AddNode(Volume1,1,transf3);              

   // Voxelize "multi_union"
   multi_union -> Voxelize();

   cout << "[> DisplayVoxelLimits:" << endl;   
   multi_union -> fVoxels -> DisplayVoxelLimits();   

   cout << "[> DisplayBoundaries:" << endl;      
   multi_union -> fVoxels -> DisplayBoundaries();   

   cout << "[> BuildListNodes:" << endl;      
   multi_union -> fVoxels -> DisplayListNodes();


   const UVector3 testPoint(0,-300,0);
   cout << endl;
   cout << "----------" << endl;
   cout << "testPoint: [" << testPoint.x << " , " << testPoint.y << " , " << testPoint.z << "]" << endl;
   cout << "----------" << endl << endl;

   cout << "[> Test Inside:" << endl;      
   VUSolid::EnumInside isInside;
   isInside = multi_union->Inside(testPoint);
   
   if(isInside == VUSolid::eInside)
   {
      cout << "    INSIDE" << endl;
   }
   else if(isInside == VUSolid::eOutside)
   {
      cout << "    OUTSIDE" << endl;
   }
   else if(isInside == VUSolid::eSurface)
   {
      cout << "    SURFACE" << endl;
   }

   cout << "[> Test Normal:" << endl;  
   UVector3 resultNormal;
   bool boolNormal;
      
   boolNormal = multi_union->Normal(testPoint,resultNormal);
   
//   if(boolNormal == true)
   {
      cout << "Normal vector: [" << resultNormal.x << " , " << resultNormal.y << " , " << resultNormal.z << " ]" << endl;
   }  
     
   cout << "[> Test Capacity:" << endl;
   double outcomeCapacity = multi_union->Capacity();
   cout << "Computed capacity: " << outcomeCapacity << endl;
  
   cout << "[> Test SafetyFromInside:" << endl;   
   double outcomeSafetyFromInside = multi_union->SafetyFromInside(testPoint,true);
   cout << "Computed SafetyFromInside: " << outcomeSafetyFromInside << endl;    

   cout << "[> Test SafetyFromOutside:" << endl;
   double outcomeSafetyFromOutside = multi_union->SafetyFromOutside(testPoint,true);
   cout << "Computed SafetyFromOutside: " << outcomeSafetyFromOutside << endl;
   
   cout << "[> Test DistanceToIn:" << endl;
   const UVector3 testDirection(0,1,0);  
   cout << "----------" << endl;
   cout << "testDirection: [" << testDirection.x << " , " << testDirection.y << " , " << testDirection.z << "]" << endl;
   cout << "----------" << endl;  
   double outcomeDistanceToIn = multi_union->DistanceToIn(testPoint,testDirection.Unit(),500.);
   cout << "Computed DistanceToIn: " << outcomeDistanceToIn << endl;        
 
   // Draw structure:
   geom->GetTopVolume()->Draw();
  
   // Program comes to an end:
   printf("[> END\n");
}

