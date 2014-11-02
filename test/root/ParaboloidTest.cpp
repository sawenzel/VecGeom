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
#include "TMath.h"
#include "TF1.h"


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
    
    GeoManager::Instance().SetWorld(worldPlaced);
    //Vector<Daughter> dau=worldPlaced->daughters();
    std::cout<<"World and paraboloid placed\n";

    //My placed volume
    //dau[0]->PrintContent();
    
    VPlacedVolume *paraboloidPlaced=paraboloid.Place();
    paraboloidPlaced->PrintContent();

    
    int np=10000, //10^4
    myCountIn=0,
    myCountOut=0,
    rootCountIn=0,
    rootCountOut=0,
    mismatchDistToIn=0,
    mismatchDistToOut=0,
    mismatchSafetyToIn=0,
    mismatchSafetyToOut=0,
    unvalidatedSafetyToIn=0,
    unvalidatedSafetyToOut=0,
    notValidSafetyToIn=0,
    notValidSafetyToOut=0;
    
    float mbDistToIn,
    rootDistToIn,
    mbDistToOut,
    rootDistToOut,
    mbSafetyToOut,
    rootSafetyToOut,
    mbSafetyToIn,
    rootSafetyToIn;
    
    
    
    double coord[3], direction[3], new_coord[3], module,
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
    
    
    //Marker for inside points
    TPolyMarker3D *markerInside=0;
    TObjArray *pm = new TObjArray(128);
    markerInside = (TPolyMarker3D*)pm->At(4);
    markerInside = new TPolyMarker3D();
    markerInside->SetMarkerColor(kYellow);
    markerInside->SetMarkerStyle(8);
    markerInside->SetMarkerSize(0.4);
    pm->AddAt(markerInside, 4);
    
    //Marker for outside points
    TPolyMarker3D *markerOutside=0;
    markerOutside = (TPolyMarker3D*)pm->At(4);
    markerOutside = new TPolyMarker3D();
    markerOutside->SetMarkerColor(kGreen+1);
    markerOutside->SetMarkerStyle(8);
    markerOutside->SetMarkerSize(0.1);
    pm->AddAt(markerOutside, 4);
    
    
    //Marker for sphere outside points
    TPolyMarker3D *markerSphereOutside=0;
    markerSphereOutside = (TPolyMarker3D*)pm->At(4);
    
    //Marker for sphere inside points
    TPolyMarker3D *markerSphereInside=0;
    markerSphereInside = (TPolyMarker3D*)pm->At(4);
    int counter;
    
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
        
        //the point is outside!
        if(inside==0){
            rootCountOut++;
            
            //mbDistToIn=dau[0]->DistanceToIn(points[i], dir[i]);
            
            
            //DISTANCE TO IN
            mbDistToIn=paraboloidPlaced->DistanceToIn(points[i], dir[i]);
            rootDistToIn=par->DistFromOutside(coord, direction);
            if( (mbDistToIn!=rootDistToIn) && !(mbDistToIn == kInfinity))
            {
                //markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
                std::cout<<"mbDistToIn: "<<mbDistToIn;
                std::cout<<" rootDistToIn: "<<rootDistToIn<<"\n";
                mismatchDistToIn++;
            }
            
            //SAFET TO IN
            mbSafetyToIn=paraboloidPlaced->SafetyToIn(points[i]);
            rootSafetyToIn=par->Safety(coord, false);
            
            //validation of SafetyToIn
            //I shoot random point belonging to the sphere with radious mbSafetyToIn and
            //then I see it they are all still outside the volume
            
            markerSphereOutside = new TPolyMarker3D();
            markerSphereOutside->SetMarkerColor(kGreen+i);
            counter=0;
            for (int j=0; j<100000; j++) //10^5
            {
            
                double v=r3.Uniform(0, 1);
                double theta=r3.Uniform(0, 2*kPi);
                double phi=TMath::ACos(2*v-1);
                
                double r= mbSafetyToIn*TMath::Power(r3.Uniform(0, 1), 1./3);
                //std::cout<<"r: "<<r<<"\n";
                
                
                double x_offset=r*TMath::Cos(theta)*TMath::Sin(phi);
                double y_offset=r*TMath::Sin(theta)*TMath::Sin(phi);
                
                double z_offset=r*TMath::Cos(phi);
                
                new_coord[0]=coord[0]+x_offset;
                new_coord[1]=coord[1]+y_offset;
                new_coord[2]=coord[2]+z_offset;
                
                double safety2=mbSafetyToIn*mbSafetyToIn;
                
                if(x_offset*x_offset+y_offset*y_offset+z_offset*z_offset<=safety2)
                {
                    counter++;
                    markerSphereOutside->SetNextPoint(new_coord[0], new_coord[1], new_coord[2]);
                    inside=someVolume->Contains(new_coord);
                    if(inside) notValidSafetyToIn++;
                }
                
            }
	    //if (markerSphereOutside) markerSphereOutside->Draw("SAME");
		//c->Update();
		
            
            if( (mbSafetyToIn!=rootSafetyToIn))
            {
                //std::cout<<"mbSafetyToIn: "<<mbSafetyToIn;
                //std::cout<<" rootSafetyToIn: "<<rootSafetyToIn<<"\n";
                mismatchSafetyToIn++;
            }
            if( (mbSafetyToIn>rootSafetyToIn))
            {
                //std::cout<<"mbSafetyToIn: "<<mbSafetyToIn;
                //std::cout<<" rootSafetyToIn: "<<rootSafetyToIn<<"\n";
                unvalidatedSafetyToIn++;
            }
            
        }
        else{
            
            //POINT IS INSIDE
            rootCountIn++;
            
            //DISTANCE TO OUT
            mbDistToOut=paraboloidPlaced->DistanceToOut(points[i], dir[i]);
            rootDistToOut=par->DistFromInside(coord, direction);
            if( (mbDistToOut!=rootDistToOut))
            {
                //markerOutside->SetNextPoint(points[i].x(), points[i].y(), points[i].z());
                std::cout<<"mbDistToOut: "<<mbDistToOut;
                std::cout<<" rootDistToOut: "<<rootDistToOut<<"\n";
                mismatchDistToOut++;
            }
            
            
            //SAFETY TO OUT
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
            
            //validation of SafetyToOut
            //I shoot random point belonging to the sphere with radious mbSafetyToOut and
            //then I see it they are all still outside the volume
            
            markerSphereInside = new TPolyMarker3D();
            markerSphereInside->SetMarkerColor(kGreen+i);
            for (int j=0; j<10000; j++)
            {
                
                double v=r3.Uniform(0, 1);
                double theta=r3.Uniform(0, 2*kPi);
                double phi=TMath::ACos(2*v-1);
                
                double r= mbSafetyToOut*TMath::Power(r3.Uniform(0, 1), 1./3);
                
                double x_offset=r*TMath::Cos(theta)*TMath::Sin(phi);
                double y_offset=r*TMath::Sin(theta)*TMath::Sin(phi);
                
                double z_offset=r*TMath::Cos(phi);
                
                new_coord[0]=coord[0]+x_offset;
                new_coord[1]=coord[1]+y_offset;
                new_coord[2]=coord[2]+z_offset;
                
                double safety2=mbSafetyToOut*mbSafetyToOut;
                
                if(x_offset*x_offset+y_offset*y_offset+z_offset*z_offset<=safety2)
                {
                    markerSphereInside->SetNextPoint(new_coord[0], new_coord[1], new_coord[2]);
                    inside=someVolume->Contains(new_coord);
                    if(!inside) notValidSafetyToOut++;
                }
                
            }
        }
    }
    
    
    //if (markerInside) markerInside->Draw("SAME");
    //c->Update();
    //sleep(3);
    
    //if (markerOutside) markerOutside->Draw("SAME");
    //c->Update();
    //sleep(3);
    
    std::cout<<"MB: NPointsInside: "<<myCountIn<<" NPointsOutside: "<<myCountOut<<" \n";
    std::cout<<"Root: NPointsInside: "<<rootCountIn<<" NPointsOutside: "<<rootCountOut<<" \n";
    std::cout<<"DistToIn mismatches: "<<mismatchDistToIn<<" \n";
    std::cout<<"DistToOut mismatches: "<<mismatchDistToOut<<" \n";
    std::cout<<"SafetyToIn mismatches: "<<mismatchSafetyToIn<<" \n";
    std::cout<<"SafetyToOut mismatches: "<<mismatchSafetyToOut<<" \n";
    std::cout<<"Against ROOT unvalidated SafetyToIn: "<<unvalidatedSafetyToIn<<" \n";
    std::cout<<"Against ROOT Unvalidated SafetyToOut: "<<unvalidatedSafetyToOut<<" \n";
    std::cout<<"Not valid SafetyToIn: "<<notValidSafetyToIn<<" \n";
    std::cout<<"Not valid SafetyToOut: "<<notValidSafetyToOut<<" \n";
    std::cout<<"Counter: "<<counter<<" \n";
    
    theApp.Run();
    

    return 0;
}

