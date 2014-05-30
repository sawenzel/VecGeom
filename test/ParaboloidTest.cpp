/// @file ParaboloidTest.cpp
/// @author Marilena Bandieramonte (marilena.bandieramonte@cern.ch)

#include "volumes/logical_volume.h"
#include "volumes/box.h"
#include "volumes/Paraboloid.h"
#include "benchmarking/Benchmarker.h"
#include "management/geo_manager.h"


#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoParaboloid.h"
#include "TGraph2D.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoParaboloid.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAttMarker.h"
#endif


using namespace vecgeom;

int main( int argc,  char *argv[]) {
    
#ifdef VECGEOM_ROOT
    
    TApplication theApp("App",&argc,argv);
    
    
    if (argc<3)
	{
    	std::cout << "usage " << argv[0] << " <Rlo[0-10]> <Rhi[0-10]> <dZ[0-10]>\n";
    	return 0;
 	}
   	
    
    double rlo=atoi(argv[1]), rhi=atoi(argv[2]), dz=atoi(argv[3]);
    
    std::cout<<"Setting up the geometry\n";
    UnplacedBox worldUnplaced = UnplacedBox(10., 10., 10.);
    UnplacedParaboloid paraboloidUnplaced = UnplacedParaboloid(rlo, rhi, dz);
    std::cout<<"World and paraboloid unplaced created\n";
    LogicalVolume world = LogicalVolume("MBworld", &worldUnplaced);
    LogicalVolume paraboloid = LogicalVolume("paraboloid", &paraboloidUnplaced);
    world.PlaceDaughter(&paraboloid, &Transformation3D::kIdentity);
    VPlacedVolume *worldPlaced = world.Place();
    //worldPlaced->PrintContent();
    
    GeoManager::Instance().set_world(worldPlaced);
    Vector<Daughter> dau=worldPlaced->daughters();
    std::cout<<"World and paraboloid placed\n";

    //My placed volume
    dau[0]->PrintContent();
    
    //VPlacedVolume *paraboloidPlaced=paraboloid.Place();
    //paraboloidPlaced->PrintContent();

    
    int np=1000000,
    myCountIn=0,
    myCountOut=0,
    rootCountIn=0,
    rootCountOut=0;
    
    double coord[3],
    x=worldUnplaced.x(),
    y=worldUnplaced.y(),
    z=worldUnplaced.z();
    
    bool inside;
    
    Vector3D <Precision> *points = new Vector3D<Precision>[np];
    TRandom3 r3;
    r3.SetSeed(time(NULL));
    
    for(int i=0; i<np; i++) // points inside world volume
    {
        points[i].x()=r3.Uniform(-x, x);
        points[i].y()=r3.Uniform(-y, y);
        points[i].z()=r3.Uniform(-z, z);
    }
    
    new TGeoManager("world", "the simplest geometry");
    TGeoMaterial *mat = new TGeoMaterial("Vacuum",0,0,0);
    TGeoMedium *med = new TGeoMedium("Vacuum",1,mat);
    TGeoVolume *top = gGeoManager->MakeBox("Top",med,worldUnplaced.x(),worldUnplaced.y(),worldUnplaced.z());
    
    
    gGeoManager->SetTopVolume(top);
    gGeoManager->CloseGeometry();
    top->SetLineColor(kMagenta);
    gGeoManager->SetTopVisible();

    TGeoVolume *someVolume = gGeoManager->MakeParaboloid("myParab", med, paraboloidUnplaced.GetRlo(), paraboloidUnplaced.GetRhi(), paraboloidUnplaced.GetDz());
   // TGeoParaboloid *par=new TGeoParaboloid("myParab", med, 3., 5., 7.);

    top->AddNode(someVolume,1);
    TCanvas *c=new TCanvas();
    top->Draw();
    sleep(3);
    c->Update();
    sleep(3);
    
    TPolyMarker3D *markerInside=0;
    TObjArray *pm = new TObjArray(128);
    markerInside = (TPolyMarker3D*)pm->At(4);
    markerInside = new TPolyMarker3D();
    markerInside->SetMarkerColor(kYellow);
    markerInside->SetMarkerStyle(8);
    markerInside->SetMarkerSize(0.4);
    pm->AddAt(markerInside, 4);
    
    
    TPolyMarker3D *markerOutside=0;
    TObjArray *pm1 = new TObjArray(128);
    markerOutside = (TPolyMarker3D*)pm->At(4);
    markerOutside = new TPolyMarker3D();
    
    
    TColor *col4 = gROOT->GetColor(4);
    col4->SetAlpha(0.01);
    
    markerOutside->SetMarkerColor(kGreen+1);
    markerOutside->SetMarkerStyle(8);
    markerOutside->SetMarkerSize(0.1);
    pm1->AddAt(markerOutside, 4);
    
    for(int i=0; i<np; i++)
    {
        inside=dau[0]->Inside(points[i]);
        if(inside==0){
            myCountOut++;
            markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
            
        }
        else{
            myCountIn++;
            markerInside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
        }
        coord[0]=points[i].x();
        coord[1]=points[i].y();
        coord[2]=points[i].z();
        
        inside=someVolume->Contains(coord);
        if(inside==0){
            rootCountOut++;
            //
            //markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
            
        }
        else{
            rootCountIn++;
            //markerInside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
        }

    }

    if (markerInside) markerInside->Draw("SAME");
    c->Update();
    sleep(3);
    if (markerOutside) markerOutside->Draw("SAME");
    c->Update();
    sleep(3);
    std::cout<<"MB:  NPointsInside: "<<myCountIn<<" NPointsOutside: "<<myCountOut<<" \n";
    std::cout<<"Root: NPointsInside: "<<rootCountIn<<" NPointsOutside: "<<rootCountOut<<" \n";

    theApp.Run();
    
#endif
    return 0;
}

