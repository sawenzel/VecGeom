#include "UBox.hh"
#include "UMultiUnion.hh"
#include "TGeoUShape.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"

void TestMultiUnion()
{
   // Initialization of ROOT environment:
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
   UBox *box1 = new UBox("UBox",50,50,50);
   UBox *box2 = new UBox("UBox",60,60,60);
   double *transform1 = new double[3];
   transform1[0] = 0;
   transform1[1] = 0;
   transform1[2] = 0;
   double *transform2 = new double[3];
   transform2[0] = 0;
   transform2[1] = 200;
   transform2[2] = 0; 
      // Constructor:
   UMultiUnion *multi_union = new UMultiUnion("multi_union");

   UMultiUnion::UNode* node1 = new UMultiUnion::UNode(box1,transform1);   
   UMultiUnion::UNode* node2 = new UMultiUnion::UNode(box2,transform2);
   
   multi_union->AddNode(node1);
   multi_union->AddNode(node2);   
   
   geom->CloseGeometry();   
   
   // Creation of a test point:
   UVector3 test_point;
   test_point.x = 700;
   test_point.y = 700;
   test_point.z = 700;   
   
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

   // Test of method SafetyFromInside:
   UVector3 TestPoint;    
   TestPoint.x = 0;
   TestPoint.y = 200;
   TestPoint.z = 0;   
   double result_safety = multi_union->UMultiUnion::SafetyFromInside(TestPoint,false);
   printf("[> SafetyFromInside:\n");
   printf("%f\n",result_safety);
   
   // Program comes to an end:
   printf("[> END\n");
}
