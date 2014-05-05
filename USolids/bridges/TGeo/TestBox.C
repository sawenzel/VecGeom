#include "UBox.hh"
#include "TGeoUShape.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"

void TestBox() {
// Make a simple geometry containing a UBox and check it
// Load as .L TestBox.C+
   TGeoManager *geom = new TGeoManager("UBox", "test of a UBox");
   TGeoMaterial *matVacuum = new TGeoMaterial("Vacuum", 0,0,0);
   TGeoMaterial *matAl = new TGeoMaterial("Al", 26.98,13,2.7);
   TGeoMedium *Vacuum = new TGeoMedium("Vacuum",1, matVacuum);
   TGeoMedium *Al = new TGeoMedium("Root Material",2, matAl);

   TGeoVolume *top = geom->MakeBox("TOP", Vacuum, 200., 200., 200.);
   TGeoShape *tgeobox = top->GetShape();
   geom->SetTopVolume(top);
   
   UBox *box = new UBox("UBox", 50,50,50);
   TGeoUShape *shape = new TGeoUShape(box->GetName(), box);
   
   TGeoVolume *vol = new TGeoVolume("UBox", shape, Al);
   top->AddNode(vol,1);
   geom->CloseGeometry();
   printf("#### Test #1: distances ###");
   printf("TGeoBBox:\n");
   tgeobox->CheckShape(1);
   printf("UBox:\n");
   shape->CheckShape(1);
   printf("#### Test #2: safety ###");
   printf("TGeoBBox:\n");
   tgeobox->CheckShape(2);
   printf("UBox:\n");   
   shape->CheckShape(2);
   printf("#### Test #3: normals ###");
   printf("TGeoBBox:\n");
   tgeobox->CheckShape(3);
   printf("UBox:\n");
   shape->CheckShape(3);
}
   
  
   
