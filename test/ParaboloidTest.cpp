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
#endif


using namespace vecgeom;

int main( int argc,  char *argv[]) {
    
#ifdef VECGEOM_ROOT
    
    TApplication theApp("App",&argc,argv);
    
#endif

    std::cout<<"Paraboloid Benchmark\n";
    UnplacedBox worldUnplaced = UnplacedBox(18., 11., 11.);
    UnplacedParaboloid paraboloidUnplaced = UnplacedParaboloid(3., 5., 7.); //rlo=3. - rhi=5. dz=7
    std::cout<<"Paraboloid created\n";
    LogicalVolume world = LogicalVolume("MBworld", &worldUnplaced);
    LogicalVolume paraboloid = LogicalVolume("paraboloid", &paraboloidUnplaced);
    world.PlaceDaughter(&paraboloid, &Transformation3D::kIdentity);
    VPlacedVolume *worldPlaced = world.Place();
    //worldPlaced->PrintContent();
    
    GeoManager::Instance().set_world(worldPlaced);
    Vector<Daughter> dau=worldPlaced->daughters();
    
    //My placed volume
    dau[0]->PrintContent();
    
    std::cout<<"World set\n";
    
    VPlacedVolume *paraboloidPlaced=paraboloid.Place();
    //paraboloidPlaced->PrintContent();
    
#ifdef VECGEOM_ROOT
    int np=100000, myCountIn=0, myCountOut=0, rootCountIn=0, rootCountOut=0;
    double coord[3];
    bool inside;
    Vector3D <Precision> *points = new Vector3D<Precision>[np];
    TRandom3 r3;
    r3.SetSeed(time(NULL)+688);
    
    for(int i=0; i<np; i++) // points inside world volume
    {
        
        points[i].x()=r3.Uniform(-10, 10);
        points[i].y()=r3.Uniform(-10, 10);;
        points[i].z()=r3.Uniform(-10, 10);;
    }
    
    new TGeoManager("world", "the simplest geometry");
    TGeoMaterial *mat = new TGeoMaterial("Vacuum",0,0,0);
    TGeoMedium *med = new TGeoMedium("Vacuum",1,mat);
    TGeoVolume *top = gGeoManager->MakeBox("Top",med,10.,10.,10.);
    
    
    gGeoManager->SetTopVolume(top);
    gGeoManager->CloseGeometry();
    top->SetLineColor(kMagenta);
    gGeoManager->SetTopVisible();
    
    TGeoVolume *someVolume = gGeoManager->MakeParaboloid("myParab", med, 3., 5., 7.);
   // TGeoParaboloid *par=new TGeoParaboloid("myParab", med, 3., 5., 7.);

    top->AddNode(someVolume,1);
    
    TCanvas *c=new TCanvas();
    top->Draw();
    c->Update();
    sleep(3);
    
    TPolyMarker3D *markerInside=0;
    TObjArray *pm = new TObjArray(128);
    markerInside = (TPolyMarker3D*)pm->At(4);
    markerInside = new TPolyMarker3D();
    markerInside->SetMarkerColor(8);
    markerInside->SetMarkerStyle(8);
    markerInside->SetMarkerSize(0.4);
    pm->AddAt(markerInside, 4);
    
    
    TPolyMarker3D *markerOutside=0;
    TObjArray *pm1 = new TObjArray(128);
    markerOutside = (TPolyMarker3D*)pm->At(4);
    markerOutside = new TPolyMarker3D();
    markerOutside->SetMarkerColor(4);
    markerOutside->SetMarkerStyle(8);
    markerOutside->SetMarkerSize(0.1);
    //markerOutside->SetMarkerTransparency(4080);
    pm1->AddAt(markerOutside, 4);
    
    
    for(int i=0; i<np; i++)
    {
        //inside=paraboloidPlaced->Inside(points[i]);
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
            //marker->SetMarkerColor(4);
            //marker->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
            
        }
        else{
            rootCountIn++;
            
            //marker->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
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

    
#endif
    
    /*Benchmarker tester(GeoManager::Instance().world());
    tester.SetVerbosity(3);
    tester.SetPointCount(1<<13);
    std::cout<<"Prepared to run benchmarker\n";
    tester.RunBenchmark();*/
    
#ifdef VECGEOM_ROOT
    
    theApp.Run();
    
#endif
    return 0;
}

