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
#include "TGeoVolume.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAttMarker.h"



using namespace vecgeom;

int main( int argc,  char *argv[]) {
    

    
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
    //Vector<Daughter> dau=worldPlaced->daughters();
    std::cout<<"World and paraboloid placed\n";

    //My placed volume
    //dau[0]->PrintContent();
    
    VPlacedVolume *paraboloidPlaced=paraboloid.Place();
    paraboloidPlaced->PrintContent();

    
    int np=1000000,
    myCountIn=0,
    myCountOut=0,
    rootCountIn=0,
    rootCountOut=0,
    mismatchDistToIn=0,
    mismatchDistToOut=0,
    mismatchSafetyToIn=0,
    mismatchSafetyToOut=0,
    unvalidatedSafetyToIn=0,
    unvalidatedSafetyToOut=0;
    
    float mbDistToIn,
    rootDistToIn,
    mbDistToOut,
    rootDistToOut,
    mbSafetyToOut,
    rootSafetyToOut,
    mbSafetyToIn,
    rootSafetyToIn;
    
    
    
    double coord[3], direction[3], module,
    x=worldUnplaced.x(),
    y=worldUnplaced.y(),
    z=worldUnplaced.z();
    
    bool inside;
    
    Vector3D <Precision> *points = new Vector3D<Precision>[np];
    Vector3D <Precision> *dir = new Vector3D<Precision>[np];
    TRandom3 r3;
    r3.SetSeed(time(NULL));
    
    int generation=0;
    
    for(int i=0; i<np; i++) // points inside world volume
    {
        
        //generation=i%5;
        
        //generic generation
        if (generation==0) {
        
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=r3.Uniform(-1, 1);
            dir[i].y()=r3.Uniform(-1, 1);
            dir[i].z()=r3.Uniform(-1, 1);
            
        }
        
        //points generated everywhere and directions pointing to the origin
        if (generation==1) {
            
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=-points[i].z();
        
        }
        
        //points generated everywhere and directions perpendicular to the z-axix
        if (generation==2) {
            
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=0;
        }
        
        //points generated in -dZ<z<dZ and directions pointing to the origin --> approaching the paraboloid from the parabolic surface
        if (generation==3) {
            
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, dz);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=0;
        }
        
        
        //points generated in -dZ<z<dZ and directions perpendicular to the z-axix --> approaching the paraboloid from the parabolic surface
        if (generation==4) {
            
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, dz);
            
            dir[i].x()=-points[i].x();
            dir[i].y()=-points[i].y();
            dir[i].z()=0;
        }
        
    
        //points outside the volume z>dZ and directions distancing from the volume
        if (generation==5) {
            
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(0, y);
            points[i].z()=r3.Uniform(dz, z);
            
            dir[i].x()=0;
            dir[i].y()=1;
            dir[i].z()=1;
        }
        
        //z>dz && uz>0 --> leaving the volume
        if (generation==6) {
            
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(dz, z);
            dir[i].x()=r3.Uniform(-1, 1);
            dir[i].y()=r3.Uniform(-1, 1);
            dir[i].z()=1;
        }
        
        //z<-dz && uz<0 --> leaving the volume
        if (generation==7) {
            
            points[i].x()=r3.Uniform(-x, x);
            points[i].y()=r3.Uniform(-y, y);
            points[i].z()=r3.Uniform(-dz, -z);
            dir[i].x()=r3.Uniform(-1, 1);
            dir[i].y()=r3.Uniform(-1, 1);
            dir[i].z()=-1;
            
        }
        
        //x^2+y^2>rhi^2 && x*ux>0 || y*uy>0, x,y>0 --> leaving the volume
        if (generation==8) {
            
            points[i].x()=r3.Uniform(rhi, x);
            points[i].y()=r3.Uniform(rhi, y);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=r3.Uniform(0, 1);
            dir[i].y()=r3.Uniform(0, 1);
            dir[i].z()=r3.Uniform(-1, 1);
            
        }
        //x^2+y^2>rhi^2 && x*ux>0 || y*uy>0, x,y<0 --> leaving the volume
        if (generation==9) {
            
            points[i].x()=r3.Uniform(-rhi, -x);
            points[i].y()=r3.Uniform(-rhi, -y);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=r3.Uniform(0, -1);
            dir[i].y()=r3.Uniform(0, -1);
            dir[i].z()=r3.Uniform(-1, 1);
            
        }
        //x^2+y^2>rhi^2 && x*ux<0 || y*uy<0 --> approaching the volume
        if (generation==10) {
            
            points[i].x()=r3.Uniform(-x, -rhi);
            points[i].y()=r3.Uniform(0, rhi);
            points[i].z()=r3.Uniform(-z, z);
            
            dir[i].x()=r3.Uniform(-1, 0);
            dir[i].y()=r3.Uniform(0, 1);
            dir[i].z()=r3.Uniform(-1, 1);
            
        }
        //x^2+y^2>rhi^2 && x*ux<0 || y*uy<0 --> approaching the volume
        if (generation==11) {
            
            points[i].x()=r3.Uniform(rhi, x);
            points[i].y()=r3.Uniform(-y, -rhi);
            points[i].z()=r3.Uniform(-z, z);
            dir[i].x()=r3.Uniform(0, 1);
            dir[i].y()=r3.Uniform(-1, 0);
            dir[i].z()=r3.Uniform(-1, 1);
            
        }
        //hitting dz surface
        if (generation==10) {
         
            float distZ, xHit, yHit, rhoHit2;
            points[i].z()=r3.Uniform(dz, z);
            dir[i].x()=-1;
            dir[i].y()=-1;
            do{
                points[i].x()=r3.Uniform(0, x);
                points[i].y()=r3.Uniform(0, y);
                dir[i].z()=r3.Uniform(-1, 0);
                distZ = (Abs(points[i].z())-dz)/Abs(dir[i].z());
                xHit = points[i].x()+distZ*dir[i].x();
                yHit = points[i].y()+distZ*dir[i].y();
                rhoHit2=xHit*xHit+yHit*yHit;
            }
            while (rhoHit2>rhi*rhi);
        
        }
        
        //hitting -dz surface
        if (generation==11) {
            
            float distZ, xHit, yHit, rhoHit2;
            points[i].z()=r3.Uniform(-dz, -z);
            dir[i].x()=-1;
            dir[i].y()=-1;
            do{
                points[i].x()=r3.Uniform(-x, x);
                points[i].y()=r3.Uniform(-y, y);
                dir[i].z()=r3.Uniform(0, 1);
                distZ = (Abs(points[i].z())-dz)/Abs(dir[i].z());
                xHit = points[i].x()+distZ*dir[i].x();
                yHit = points[i].y()+distZ*dir[i].y();
                rhoHit2=xHit*xHit+yHit*yHit;
            }
            while (rhoHit2>rlo*rlo);
            
        }
        
        
        module=Sqrt(dir[i].x()*dir[i].x()+dir[i].y()*dir[i].y()+dir[i].z()*dir[i].z());
        dir[i].x()=dir[i].x()/module;
        dir[i].y()=dir[i].y()/module;
        dir[i].z()=dir[i].z()/module;
        
    
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
    TGeoParaboloid *par=new TGeoParaboloid("myParab", paraboloidUnplaced.GetRlo(), paraboloidUnplaced.GetRhi(), paraboloidUnplaced.GetDz());

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
        inside=paraboloidPlaced->Inside(points[i]);
        if(inside!=0){ //Enum-inside give back 0 if the point is inside 
            
            
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
        
        direction[0]=dir[i].x();
        direction[1]=dir[i].y();
        direction[2]=dir[i].z();
        
        
        inside=someVolume->Contains(coord);
        //inside=par->Contains(coord);
        if(inside==0){
            rootCountOut++;
            
            //mbDistToIn=dau[0]->DistanceToIn(points[i], dir[i]);
            mbDistToIn=paraboloidPlaced->DistanceToIn(points[i], dir[i]);
            rootDistToIn=par->DistFromOutside(coord, direction);
            if( (mbDistToIn!=rootDistToIn) && !(mbDistToIn == kInfinity))
            {
                //markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
                std::cout<<"mbDistToIn: "<<mbDistToIn;
                std::cout<<" rootDistToIn: "<<rootDistToIn<<"\n";
                mismatchDistToIn++;
            }
            
            mbSafetyToIn=paraboloidPlaced->SafetyToIn(points[i]);
            rootSafetyToIn=par->Safety(coord, false);
            if( (mbSafetyToIn!=rootSafetyToIn))
            {
                //std::cout<<"mbSafetyToIn: "<<mbSafetyToIn;
                //std::cout<<" rootSafetyToIn: "<<rootSafetyToIn<<"\n";
                mismatchSafetyToIn++;
            }
            if( (mbSafetyToIn>rootSafetyToIn))
            {
                std::cout<<"mbSafetyToIn: "<<mbSafetyToIn;
                std::cout<<" rootSafetyToIn: "<<rootSafetyToIn<<"\n";
                unvalidatedSafetyToIn++;
            }
            
        }
        else{
            rootCountIn++;
            mbDistToOut=paraboloidPlaced->DistanceToOut(points[i], dir[i]);
            rootDistToOut=par->DistFromInside(coord, direction);
            if( (mbDistToOut!=rootDistToOut))
            {
                //markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
                std::cout<<"mbDistToOut: "<<mbDistToOut;
                std::cout<<" rootDistToOut: "<<rootDistToOut<<"\n";
                mismatchDistToOut++;
            }
            
            mbSafetyToOut=paraboloidPlaced->SafetyToOut(points[i]);
            rootSafetyToOut=par->Safety(coord, true);
            if( (mbSafetyToOut!=rootSafetyToOut))
            {
                //std::cout<<"mbSafetyToOut: "<<mbSafetyToOut;
                //std::cout<<" rootSafetyToOut: "<<rootSafetyToOut<<"\n";
                mismatchSafetyToOut++;
            }
            if( (mbSafetyToOut>rootSafetyToOut))
            {
                unvalidatedSafetyToOut++;
            }
        }

    }
    

    if (markerInside) markerInside->Draw("SAME");
    c->Update();
    sleep(3);
    if (markerOutside) markerOutside->Draw("SAME");
    c->Update();
    sleep(3);
    
    std::cout<<"MB: NPointsInside: "<<myCountIn<<" NPointsOutside: "<<myCountOut<<" \n";
    std::cout<<"Root: NPointsInside: "<<rootCountIn<<" NPointsOutside: "<<rootCountOut<<" \n";
    std::cout<<"DistToIn mismatches: "<<mismatchDistToIn<<" \n";
    std::cout<<"DistToOut mismatches: "<<mismatchDistToOut<<" \n";
    std::cout<<"SafetyToIn mismatches: "<<mismatchSafetyToIn<<" \n";
    std::cout<<"SafetyToOut mismatches: "<<mismatchSafetyToOut<<" \n";
    std::cout<<"Unvalidated SafetyToIn: "<<unvalidatedSafetyToIn<<" \n";
    std::cout<<"Unvalidated SafetyToOut: "<<unvalidatedSafetyToOut<<" \n";
    
    theApp.Run();
    

    return 0;
}

