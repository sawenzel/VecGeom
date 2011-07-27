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
   UBox *box = new UBox("UBox",40,40,40);

   TGeoUShape *Shape1 = new TGeoUShape("Shape1",box);
   TGeoVolume *Volume1 = new TGeoVolume("Volume1",Shape1);
   Volume1->SetLineColor(1);
   
      // Number of nodes to implement:
   int numNodesImpl = 10;
   int mIndex = 0, nIndex = 0, oIndex = 0;
	int carBoxesX = 10;
	int carBoxesY = 10;
	int carBoxesZ = 10;   
   UTransform3D* arrayTransformations[numNodesImpl];
	TGeoCombiTrans* arrayCombiTrans[numNodesImpl];
  
     // Constructor:
   UMultiUnion *multi_union = new UMultiUnion("multi_union");     
   
      // Transformation:
   for(mIndex = 0 ; (mIndex < numNodesImpl) && (mIndex < carBoxesX*carBoxesY*carBoxesZ) ; mIndex++)
   {
      arrayTransformations[mIndex] = new UTransform3D(-1000+50+2*50*(mIndex%carBoxesX),-1000+50+2*50*nIndex,-1000+50+2*50*oIndex,0,0,0);
      multi_union->AddNode(box,arrayTransformations[mIndex]);
      
      arrayCombiTrans[mIndex] = new TGeoCombiTrans(-1000+50+2*50*(mIndex%carBoxesX),-1000+50+2*50*nIndex,-1000+50+2*50*oIndex,new TGeoRotation("rot",0,0,0));      
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
   int npoints = 10000000;
   TPolyMarker3D *pminside = new TPolyMarker3D();
   pminside->SetMarkerColor(kRed);
   TPolyMarker3D *pmoutside = new TPolyMarker3D();
   pmoutside->SetMarkerColor(kGreen);
   TStopwatch timer;
   timer.Start();
   int n10 = npoints/10;
   for (int ipoint = 0; ipoint<npoints; ipoint++) {
      if (n10 && (ipoint%n10)==0) printf("test inside ... %d%%\n",int(100*ipoint/npoints));
      point.x = gRandom->Uniform(bmin[0], bmax[0]);
      point.y = gRandom->Uniform(bmin[1], bmax[1]);
      point.z = gRandom->Uniform(bmin[2], bmax[2]);
      VUSolid::EnumInside inside = multi_union->Inside(point);
      if (inside==VUSolid::eInside || inside==VUSolid::eSurface) {
         pminside->SetNextPoint(point.x, point.y, point.z);
      } else {
         pmoutside->SetNextPoint(point.x, point.y, point.z);
      }   
   }
   timer.Stop();
   timer.Print();
   geom->GetTopVolume()->Draw();
   pminside->Draw();
//   pmoutside->Draw();
   return;

   // Test of GetCandidatesVoxel:
   cout << "[> GetCandidatesVoxel:" << endl;
   int selection1, selection2, selection3;
   cout << "Please enter the coordinates of the voxel to be tested, separated by commas." << endl;
   cout << "Enter coordinate -1 for first coordinate to leave." << endl;
   cout << "   [> ";
   scanf("%d,%d,%d",&selection1,&selection2,&selection3);      


   
   
   do
   {  
      if(selection1 == -1) continue;
      multi_union -> fVoxels -> GetCandidatesVoxel(selection1,selection2,selection3);   
      cout << "   [> ";
      scanf("%d,%d,%d",&selection1,&selection2,&selection3);
   }
   while(selection1 != -1);

   // Test of Inside:
      // Definition of a timer in order to compare the scalability of the two methods:       
   TStopwatch *Chronometre;
	Chronometre = new TStopwatch();     
   // Creation of a test point:   
   double coX, coY, coZ;
   cout << "[> Inside:" << endl;
   cout << "Please enter separately the coordinates of the point to be tested." << endl;
   cout << "Enter coordinate -1 for first coordinate to leave." << endl;
   cout << "   [> ";
   cin >> coX;
   cout << "   [> ";
   cin >> coY;
   cout << "   [> ";
   cin >> coZ;        
   
   UVector3 test_point;
   test_point.Set(coX,coY,coZ);
  
   do
   {  
      if(coX == -1) continue;
    	Chronometre->Reset();      
      Chronometre->Start();            
      VUSolid::EnumInside resultat = multi_union->Inside(test_point);
   	Chronometre->Stop();      

      cout << "  Tested point: [" << test_point.x << "," << test_point.y << "," << test_point.z << "]" << endl;

      if(resultat == 0)
      {
         cout << "  is INSIDE the defined solid" << endl;
      }
      else if(resultat == 1)
      {
         cout << "  is on a SURFACE of the defined solid" << endl;
      }
      else
      {
         cout << "  is OUTSIDE the defined solid" << endl;
      }
       
      cout << "Timer: ";
      Chronometre->Print();
      cout << endl;
          
      cout << "   [> ";
      cin >> coX;
      cout << "   [> ";
      cin >> coY;
      cout << "   [> ";
      cin >> coZ; 
   
      test_point.Set(coX,coY,coZ);     
   }
   while(coX != -1);  

	delete Chronometre;   
      
   // RayTracing:
   int choice = 0;
   printf("[> In order to trace the geometry, type: 1. To exit, press 0 and return:\n");
   scanf("%d",&choice);
   
   if(choice == 1)
   {
      top->Draw();
   }
   else
   {
     // Do nothing
   }
  
   // Program comes to an end:
   printf("[> END\n");
}
