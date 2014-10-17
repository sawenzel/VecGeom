/// @file ParaboloidTest.cpp
/// @author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"
#include "volumes/Paraboloid.h"
#include "benchmarking/Benchmarker.h"
#include "management/GeoManager.h"

#include "TGeoShape.h"
#include "TGeoParaboloid.h"
#include "TGraph2D.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoParaboloid.h"
#include "TGeoHype.h"
#include "TGeoVolume.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAttMarker.h"
#include "TMath.h"
#include "TF1.h"


using namespace vecgeom;

int main( int argc,  char *argv[]) {
    

    
    TApplication theApp("App",&argc,argv);
    
    if (argc<6)
	{
    	std::cout << "usage " << argv[0] << " <rIn[0-10]> <stIn[0-10]> <rOut[0-10]> <stOut[0-10]> <dZ[0-10]>\n";
    	return 0;
 	}
   	
    
    double rIn=atoi(argv[1]), stIn=atoi(argv[2]), rOut=atoi(argv[3]), stOut=atoi(argv[4]), dz=atoi(argv[5]);
    
    new TGeoManager("world", "the simplest geometry");
    TGeoMaterial *mat = new TGeoMaterial("Vacuum",0,0,0);
    TGeoMedium *med = new TGeoMedium("Vacuum",1,mat);
    TGeoVolume *top = gGeoManager->MakeBox("Top",med,10,10,10);
    
    
    gGeoManager->SetTopVolume(top);
    gGeoManager->CloseGeometry();
    top->SetLineColor(kMagenta);
    gGeoManager->SetTopVisible();


    TGeoVolume *someVolume = gGeoManager->MakeHype("myHyperboloid", med, rIn, stIn, rOut, stOut, dz);
    //TGeoHype *hype=new TGeoHype("myHype", 3., 4., 5., 6., 2.);
    
    
    

    top->AddNode(someVolume,1);
    TCanvas *c=new TCanvas();
    top->Draw();
    sleep(3);
    c->Update();
    sleep(3);
    
    theApp.Run();
    

    return 0;
}

