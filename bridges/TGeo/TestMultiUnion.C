#include "UBox.hh"
#include "UMultiUnion.hh"
#include "UTransform3D.hh"
#include "UVoxelFinder.hh"
#include "TGeoUShape.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"

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
      // Creation of two nodes:
   UBox *box1 = new UBox("UBox",100,100,100);
   UBox *box2 = new UBox("UBox",60,60,60);
   UTransform3D *transform1 = new UTransform3D(20,0,0,0,45,0);
   UTransform3D *transform2 = new UTransform3D(0,200,0,0,0,0);
      // Constructor:
   UMultiUnion *multi_union = new UMultiUnion("multi_union");
   multi_union->AddNode(box1,transform1);
   multi_union->AddNode(box2,transform2);     
   
   geom->CloseGeometry();   
   
   // Creation of a test point:
   UVector3 test_point;
   test_point.x = 600;
   test_point.y = 0;
   test_point.z = 0;   
   
   VUSolid::EnumInside resultat = multi_union->Inside(test_point);

   printf("[> Inside:\n");
      
   if(resultat == 0)
   {
      printf("test_point is INSIDE\n");
   }
   else if(resultat == 1)
   {
      printf("test_point is on a SURFACE\n");
   }
   else
   {
      printf("test_point is OUTSIDE\n");      
   }
   
   // Test of method Extent (1st version):
   VUSolid::EAxisType x_axis = VUSolid::eXaxis;
   VUSolid::EAxisType y_axis = VUSolid::eYaxis;   
   VUSolid::EAxisType z_axis = VUSolid::eZaxis;   
   
   double MinX,MaxX,MinY,MaxY,MinZ,MaxZ = 0;
   multi_union->Extent(x_axis,MinX,MaxX);
   multi_union->Extent(y_axis,MinY,MaxY);
   multi_union->Extent(z_axis,MinZ,MaxZ);
   printf("[> Extent - 1st version:\n");
   printf(" * X: [%f ; %f]\n * Y: [%f ; %f]\n * Z: [%f ; %f]\n",MinX,MaxX,MinY,MaxY,MinZ,MaxZ);
   
   // Test of method Extent (2nd version):
   double *table_mini = new double[3];
   double *table_maxi = new double[3];
 
   multi_union->Extent(table_mini,table_maxi);
   printf("[> Extent - 2nd version:\n");
   printf(" * X: [%f ; %f]\n * Y: [%f ; %f]\n * Z: [%f ; %f]\n",table_mini[0],table_maxi[0],table_mini[1],table_maxi[1],table_mini[2],table_maxi[2]);   
   
   // Test of "BuildVoxelLimits":
   printf("[> BuildVoxelLimits:\n");   
   UVoxelFinder voxfind(multi_union);
   voxfind.BuildVoxelLimits();
   voxfind.DisplayVoxelLimits();
   
   // "CreateBoundaries":
   voxfind.CreateBoundaries();
   
   // Test of "SortBoundaries":
   voxfind.SortBoundaries();
   printf("[> SortBoundaries:\n");
   voxfind.DisplayBoundaries();
   
   // Program comes to an end:
   printf("[> END\n");
}
