
#include "UBox.hh"
#include "UOrb.hh"

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

#include <fstream>
#include "UUtils.hh"

using namespace std;

const double unionMaxX = 1000.;
const double unionMaxY = 1000.;
const double unionMaxZ = 1000.;

const double extentBorder = 1.1;

const int carBoxesX = 20;
const int carBoxesY = 20;
const int carBoxesZ = 20;

UMultiUnion *CreateMultiUnion(int numNodes, TGeoVolume *top=NULL) // Number of nodes to implement
{
   // Instance:
      // Creation of several nodes:

   double extentVolume = extentBorder * 2 * unionMaxX * extentBorder * 2 * unionMaxY * extentBorder * 2 * unionMaxZ;
   double ratio = 1.0/3.0; // ratio of inside points vs (inside + outside points)
   double length = 40;
   
   if (true) length = pow (ratio * extentVolume / numNodes, 1./3.) / 2;
	
//   UBox *box = new UBox("UBox", length, length, length);

   UOrb *box = new UOrb("UOrb", length);

   TGeoVolume *volume1;
   double capacity = box->Capacity();

   if (top)
   {
		TGeoUShape *Shape1 = new TGeoUShape("Shape1",box);
		volume1 = new TGeoVolume("Volume1",Shape1);
		volume1->SetLineColor(1);
   }

   UTransform3D** arrayTransformations = new UTransform3D * [numNodes];
	TGeoCombiTrans** arrayCombiTrans = new TGeoCombiTrans * [numNodes];
  
     // Constructor:
   UMultiUnion *multiUnion = new UMultiUnion("multiUnion");

   if (true)
   {
	   for(int i = 0; i < numNodes ; i++)
	   {
		   double x = gRandom->Uniform(-unionMaxX + length, unionMaxX - length);
		   double y = gRandom->Uniform(-unionMaxY + length, unionMaxY - length);
		   double z = gRandom->Uniform(-unionMaxZ + length, unionMaxZ - length);

		   arrayTransformations[i] = new UTransform3D(x,y,z,0,0,0);
		   multiUnion->AddNode(*box,*arrayTransformations[i]);
		   arrayCombiTrans[i] = new TGeoCombiTrans(x,y,z,new TGeoRotation("rot",0,0,0));
	   }
   }
   else 
   {   
	   // Transformation:
	   for(int n = 0, o = 0, m = 0; m < numNodes ; m++)
	   {
		   if (m >= carBoxesX*carBoxesY*carBoxesZ) break;
		   double spacing = 50;
		   double x = -unionMaxX+spacing+2*spacing*(m%carBoxesX);
		   double y = -unionMaxX+spacing+2*spacing*n;
		   double z = -unionMaxX+spacing+2*spacing*o;

		  arrayTransformations[m] = new UTransform3D(x,y,z,0,0,0);
		  multiUnion->AddNode(*box,*arrayTransformations[m]);
      
		  arrayCombiTrans[m] = new TGeoCombiTrans(x,y,z,new TGeoRotation("rot",0,0,0));

		  if (top) top->AddNode(volume1, m+1, arrayCombiTrans[m]);      
      
		  // Preparing "Draw":
		  if (m % carBoxesX == carBoxesX-1)
		  {
			  if (n % carBoxesY == carBoxesY-1)
			  {
				 n = 0;
				 o++;
			  }      
			  else n++;
		  }
	   }
   }

   multiUnion->Voxelize();

   return multiUnion;
}


void DisplayMultiUnionInfo(UMultiUnion *multiUnion)
{
	cout << "[> DisplayVoxelLimits:" << endl;   
	multiUnion->GetVoxels().DisplayVoxelLimits();   

	cout << "[> DisplaySortedBoundaries:" << endl;      
	multiUnion->GetVoxels().DisplayBoundaries();   

	cout << "[> BuildListNodes:" << endl;      
	multiUnion->GetVoxels().DisplayListNodes();
}


void TestMultiUnionWithGraphics()
{
   // Initialization of ROOT environment:
   // Test for a multiple union solid.
   TGeoManager *geom = new TGeoManager("UMultiUnion","Test of a UMultiUnion");
   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
   TGeoMedium *Al = new TGeoMedium("Root Material",2, matAl);
   
   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, unionMaxX, unionMaxY, unionMaxZ);
   TGeoShape *tgeobox = top->GetShape();
   geom->SetTopVolume(top);
   
   UMultiUnion *multiUnion = CreateMultiUnion(21, top);

   DisplayMultiUnionInfo(multiUnion);

   geom->CloseGeometry();

   cout << "[> Test:" << endl;   
   
   double bmin[3], bmax[3];
   UVector3 point;
   multiUnion->Extent(bmin, bmax);
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
      VUSolid::EnumInside inside = multiUnion->Inside(point);
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

double TestMultiUnionOneStep(int n)
{
	UMultiUnion *multiUnion = CreateMultiUnion(n);
//    DisplayMultiUnionInfo(multiUnion);

    cout << "Testing with " << n << " nodes" << endl;
   
//    double bmin[3], bmax[3];
//    multiUnion->Extent(bmin, bmax);

    int npoints = 10000000;
    #ifdef DEBUG
        npoints = 10000;
    #endif

    int n10 = npoints/10;
    UVector3 point;

    TStopwatch timer;
    timer.Start();

#ifdef DEBUG
    int inside = 0, outside = 0, surface = 0;
#endif

    for (int i = 0; i < npoints; i++)
    {
        if (n > 100)
            if (n10 && (i % n10)==0) cout << "Test inside ... " << int(100*i/npoints) << "%\n";
        point.x = gRandom->Uniform(-unionMaxX * extentBorder, unionMaxX * extentBorder);
        point.y = gRandom->Uniform(-unionMaxY * extentBorder, unionMaxY * extentBorder);
        point.z = gRandom->Uniform(-unionMaxZ * extentBorder, unionMaxZ * extentBorder);

        VUSolid::EnumInside result = multiUnion->Inside(point);
        
#ifdef DEBUG
        switch (result)
        {
            case VUSolid::eInside: inside++; break;
            case VUSolid::eOutside: outside++; break;
            case VUSolid::eSurface: surface++; break;
        }
#endif
    }

    timer.Stop();
    double chrono = timer.CpuTime();

    cout << "CPUTIME: " << chrono << endl;
    return chrono;
}

void TestMultiUnion()
{
//    TestMultiUnionWithGraphics(); return;

    int numNodes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34, 36, 38, 40, 45, 50, 60, 70, 80, 90, 100, 500, 1000, 5000, 10000};
    int numNodesCount = sizeof (numNodes) / sizeof (int);
    // numNodesCount -= 3;

    ofstream nodes("nodes.dat"), times("times.dat");
    for (int i = 0; i < numNodesCount; i++)
    {
        int n = numNodes[i];
        double chrono = TestMultiUnionOneStep(n);
        nodes << n << endl;
        times << chrono << endl;
    }
    nodes.close(), times.close();
    printf("[> END\n");
}

/*
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
   UMultiUnion *multiUnion = new UMultiUnion("multiUnion");     
   UTransform3D* trans = new UTransform3D(0,0,0,0,0,0);
   UTransform3D* trans2 = new UTransform3D(200,200,0,0,0,0);      
   UTransform3D* trans3 = new UTransform3D(400,400,0,0,0,0);       
   
   multiUnion->AddNode(box,trans);
   multiUnion->AddNode(box,trans2); 
   multiUnion->AddNode(box,trans3);       
                                                                                                                                         
   geom->CloseGeometry();
   
   TGeoUShape *Shape1 = new TGeoUShape("Shape1",box);
   TGeoVolume *Volume1 = new TGeoVolume("Volume1",Shape1);
   Volume1->SetLineColor(2);   
      
   TGeoCombiTrans* transf1 = new TGeoCombiTrans(0,0,0,new TGeoRotation("rot1",0,0,0));      
   top->AddNode(Volume1,1,transf1);

   TGeoCombiTrans* transf2 = new TGeoCombiTrans(200,200,0,new TGeoRotation("rot1",0,0,0));      
   top->AddNode(Volume1,1,transf2);

   TGeoCombiTrans* transf3 = new TGeoCombiTrans(400,400,0,new TGeoRotation("rot1",0,0,0));      
   top->AddNode(Volume1,1,transf3);              

   // Voxelize "multiUnion"
   multiUnion->Voxelize();

   cout << "[> DisplayVoxelLimits:" << endl;   
   multiUnion->voxels->DisplayVoxelLimits();   

   cout << "[> DisplayBoundaries:" << endl;      
   multiUnion->voxels->DisplayBoundaries();   

   cout << "[> BuildListNodes:" << endl;      
   multiUnion->voxels->DisplayListNodes();


   const UVector3 testPoint(-220,300,0);
   cout << endl;
   cout << "----------" << endl;
   cout << "testPoint: [" << testPoint.x << " , " << testPoint.y << " , " << testPoint.z << "]" << endl;
   cout << "----------" << endl << endl;

   cout << "[> Test Inside:" << endl;      
   VUSolid::EnumInside isInside;
   isInside = multiUnion->Inside(testPoint);
   
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
      
   boolNormal = multiUnion->Normal(testPoint,resultNormal);
   
//   if(boolNormal == true)
   {
      cout << "Normal vector: [" << resultNormal.x << " , " << resultNormal.y << " , " << resultNormal.z << " ]" << endl;
   }  
     
   cout << "[> Test Capacity:" << endl;
   double outcomeCapacity = multiUnion->Capacity();
   cout << "Computed capacity: " << outcomeCapacity << endl;
  
   cout << "[> Test SafetyFromInside:" << endl;   
   double outcomeSafetyFromInside = multiUnion->SafetyFromInside(testPoint,true);
   cout << "Computed SafetyFromInside: " << outcomeSafetyFromInside << endl;    

   cout << "[> Test SafetyFromOutside:" << endl;
   double outcomeSafetyFromOutside = multiUnion->SafetyFromOutside(testPoint,true);
   cout << "Computed SafetyFromOutside: " << outcomeSafetyFromOutside << endl;
   
   cout << "[> Test DistanceToIn:" << endl;
   const UVector3 testDirection(1,0,0);  
   cout << "----------" << endl;
   cout << "testDirection: [" << testDirection.x << " , " << testDirection.y << " , " << testDirection.z << "]" << endl;
   cout << "----------" << endl;  
   double outcomeDistanceToIn = multiUnion->DistanceToIn(testPoint,testDirection,500.);
   cout << "Computed DistanceToIn: " << outcomeDistanceToIn << endl;

   cout << "[> Test DistanceToOut:" << endl;
   const UVector3 testDirectionOut(0,-1,0);
   UVector3 dtoNormal;
   bool bool1 = true;
   cout << "----------" << endl;
   cout << "testDirectionOut: [" << testDirectionOut.x << " , " << testDirectionOut.y << " , " << testDirectionOut.z << "]" << endl;
   cout << "----------" << endl;  
   double outcomeDistanceToOut = multiUnion->DistanceToOut(testPoint,testDirectionOut,dtoNormal,bool1,10000.);
   cout << "Computed DistanceToOut: " << outcomeDistanceToOut << endl;          
   cout << "NormalDistToOut: [" << dtoNormal.x << " , " << dtoNormal.y << " , " << dtoNormal.z << "]" << endl;

   // Draw structure:
   geom->GetTopVolume()->Draw();
  
   // Program comes to an end:
   printf("[> END\n");
}
*/

/*
// Test through bridge classes:
void TestMultiUnion()
{
// Make a simple geometry containing a UBox and check it
   TGeoManager *geom = new TGeoManager("UBox", "test of a UMultiUnion");
   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
   TGeoMedium *Al = new TGeoMedium("Root Material",2, matAl);

   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 1000., 1000., 1000.);
   geom->SetTopVolume(top);
   
   // Creation of several nodes
   UBox *box = new UBox("UBox",200,200,200);
  
   // Creation of the multi union
   UMultiUnion *multiUnion = new UMultiUnion("multiUnion");     
   UTransform3D* trans = new UTransform3D(0,0,0,0,0,0);
   UTransform3D* trans2 = new UTransform3D(200,200,0,0,0,0);      
   UTransform3D* trans3 = new UTransform3D(400,400,0,0,0,0);       
   
   multiUnion->AddNode(box,trans);
   multiUnion->AddNode(box,trans2); 
   multiUnion->AddNode(box,trans3);                                                                                                                                               

   // "Conversion"
   TGeoUShape *shape = new TGeoUShape("shape", multiUnion);     
   TGeoVolume *vol = new TGeoVolume("vol", shape, Al);
   top->AddNode(vol,1);
   
   // Voxelize "multiUnion"
   multiUnion->Voxelize();   
   
   geom->CloseGeometry();

   printf("#### Test #1: distances ###\n");
   printf("MultiUnion:\n");
   shape->CheckShape(1,200);

   printf("#### Test #2: safety ###\n");
   printf("MultiUnion:\n");   
   shape->CheckShape(2,200);
  
   printf("#### Test #3: normals ###\n");
   printf("MultiUnion:\n");
   shape->CheckShape(3,200);
}
*/

