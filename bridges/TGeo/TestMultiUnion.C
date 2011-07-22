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
   UBox *box1 = new UBox("UBox",80,80,80);
   UBox *box2 = new UBox("UBox",100,100,100);
   UBox *box3 = new UBox("UBox",100,100,100);
   UBox *box4 = new UBox("UBox",20,20,20);
   UBox *box5 = new UBox("UBox",50,50,50);
   UBox *box6 = new UBox("UBox",60,60,60);
   UBox *box7 = new UBox("UBox",30,60,60);
   UBox *box8 = new UBox("UBox",90,10,10);
   UBox *box9 = new UBox("UBox",40,40,40);             
   
   UTransform3D *transform1 = new UTransform3D(0,0,0,0,45,0);
   UTransform3D *transform2 = new UTransform3D(300,300,300,0,0,0);
   UTransform3D *transform3 = new UTransform3D(0,400,100,0,0,0);   
   UTransform3D *transform4 = new UTransform3D(300,500,600,10,0,0);     
   UTransform3D *transform5 = new UTransform3D(0,20,10,0,0,0);  
   UTransform3D *transform6 = new UTransform3D(-200,25,100,0,90,0);   
   UTransform3D *transform7 = new UTransform3D(0,-150,-300,70,90,30);  
   UTransform3D *transform8 = new UTransform3D(600,0,0,0,45,0);            
   UTransform3D *transform9 = new UTransform3D(0,0,0,0,20,63);                            
                    
      // Constructor:
   UMultiUnion *multi_union = new UMultiUnion("multi_union");
   multi_union->AddNode(box1,transform1);
   multi_union->AddNode(box2,transform2);
   multi_union->AddNode(box3,transform3);
   multi_union->AddNode(box4,transform4);
   multi_union->AddNode(box5,transform5);
   multi_union->AddNode(box6,transform6);
   multi_union->AddNode(box7,transform7);
   multi_union->AddNode(box8,transform8);                
   multi_union->AddNode(box9,transform9);                                     
  
   // Preparing RayTracing of the created geometry:
      // Using bridge class "TGeoUShape":
	TGeoCombiTrans *combi1 = new TGeoCombiTrans(0,0,0,new TGeoRotation("rot1",0,45,0));
	TGeoCombiTrans *combi2 = new TGeoCombiTrans(300,300,300,new TGeoRotation("rot2",0,0,0));
	TGeoCombiTrans *combi3 = new TGeoCombiTrans(0,400,100,new TGeoRotation("rot3",0,0,0));
	TGeoCombiTrans *combi4 = new TGeoCombiTrans(300,500,600,new TGeoRotation("rot4",10,0,0));
	TGeoCombiTrans *combi5 = new TGeoCombiTrans(0,20,10,new TGeoRotation("rot5",0,0,0));
	TGeoCombiTrans *combi6 = new TGeoCombiTrans(-200,25,100,new TGeoRotation("rot6",0,90,0));  
	TGeoCombiTrans *combi7 = new TGeoCombiTrans(0,-150,-300,new TGeoRotation("rot7",70,90,30));  
	TGeoCombiTrans *combi8 = new TGeoCombiTrans(600,0,0,new TGeoRotation("rot8",0,45,0));             
	TGeoCombiTrans *combi9 = new TGeoCombiTrans(0,0,0,new TGeoRotation("rot9",0,20,63));    
      
   TGeoUShape *Shape1 = new TGeoUShape("Shape1",box1);
   TGeoVolume *Volume1 = new TGeoVolume("Volume1",Shape1);
   Volume1->SetLineColor(1);
         
   TGeoUShape *Shape2 = new TGeoUShape("Shape2",box2);
   TGeoVolume *Volume2 = new TGeoVolume("Volume2",Shape2);
   Volume2->SetLineColor(2);
   
   TGeoUShape *Shape3 = new TGeoUShape("Shape3",box3);
   TGeoVolume *Volume3 = new TGeoVolume("Volume3",Shape3);   
   Volume3->SetLineColor(3);  
  
   TGeoUShape *Shape4 = new TGeoUShape("Shape4",box4);
   TGeoVolume *Volume4 = new TGeoVolume("Volume4",Shape4);   
   Volume4->SetLineColor(4);  
   
   TGeoUShape *Shape5 = new TGeoUShape("Shape5",box5);
   TGeoVolume *Volume5 = new TGeoVolume("Volume5",Shape5);   
   Volume5->SetLineColor(5);  
   
   TGeoUShape *Shape6 = new TGeoUShape("Shape6",box6);
   TGeoVolume *Volume6 = new TGeoVolume("Volume6",Shape6);   
   Volume6->SetLineColor(6);    
   
   TGeoUShape *Shape7 = new TGeoUShape("Shape7",box7);
   TGeoVolume *Volume7 = new TGeoVolume("Volume7",Shape7);   
   Volume7->SetLineColor(7);   
   
   TGeoUShape *Shape8 = new TGeoUShape("Shape8",box8);
   TGeoVolume *Volume8 = new TGeoVolume("Volume8",Shape8);   
   Volume8->SetLineColor(8);   
                                    
   TGeoUShape *Shape9 = new TGeoUShape("Shape9",box9);
   TGeoVolume *Volume9 = new TGeoVolume("Volume9",Shape9);   
   Volume9->SetLineColor(9);   
  
   top->AddNode(Volume1,1,combi1);
   top->AddNode(Volume2,2,combi2);        
   top->AddNode(Volume3,3,combi3);        
   top->AddNode(Volume4,4,combi4); 
   top->AddNode(Volume5,5,combi5); 
   top->AddNode(Volume6,6,combi6); 
   top->AddNode(Volume7,7,combi7); 
   top->AddNode(Volume8,8,combi8);            
   top->AddNode(Volume9,9,combi9);    

   geom->CloseGeometry();

   // Voxelize "multi_union"
   multi_union -> Voxelize();

   cout << "[> DisplayVoxelLimits:" << endl;   
   multi_union -> fVoxels -> DisplayVoxelLimits();   

   cout << "[> DisplayBoundaries:" << endl;      
   multi_union -> fVoxels -> DisplayBoundaries();   

   cout << "[> BuildListNodes:" << endl;      
   multi_union -> fVoxels -> DisplayListNodes();

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
