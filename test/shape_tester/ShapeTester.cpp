//
// Implementation of the batch solid  test
//

#include "base/RNG.h"

#include <iomanip>
#include <sstream>
#include <ctime>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTransform3D.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "TGeoShape.h"
#include "TGeoParaboloid.h"
#include "TGeoBBox.h"
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
#include "TH1D.h"
#include "TH2F.h"
#include "TF1.h"
#include "TVirtualPad.h"
#include "TView3D.h"
#endif
    
using namespace std;

ShapeTester::ShapeTester()
{
	SetDefaults();
}

ShapeTester::~ShapeTester()
{

}

void ShapeTester::SetDefaults()
{
	numCheckPoints = 10;
	maxPoints = 10000;
        fVerbose = 1 ;
	repeat = 1000;
	insidePercent = 100.0/3;
	outsidePercent = 100.0/3;
        edgePercent = 0;
        
	outsideMaxRadiusMultiple = 10;
	outsideRandomDirectionPercent = 50;
	differenceTolerance = 0.01;
        ifSaveAllData = true;//false;//true;
        ifMoreTests = true;//false;//true;
        ifDifUSolids = true;
        minDifference = VUSolid::Tolerance();
        difPointsInside = 0;
        difPointsSurface = 0;
        difPointsOutside = 0;

        definedNormal = false;
        ifException = false;
        maxErrorBreak =1000;

	method = "all";
	perftab = perflabels = NULL;
        volumeUSolids = NULL;

        gCapacitySampled = 0;
        gCapacityError = 0;
        gCapacityAnalytical =  0;
        gNumberOfScans = 15;


  //
  // Zero error list
  //
  errorList = 0;
      
}

UVector3 ShapeTester::GetRandomDirection() 
{
	double phi = 2.*UUtils::kPi*UUtils::Random();
	double theta = UUtils::ACos(1.-2.*UUtils::Random());
	double vx = std::sin(theta)*std::cos(phi);
	double vy = std::sin(theta)*std::sin(phi);
	double vz = std::cos(theta);
	UVector3 vec(vx,vy,vz);
	vec.Normalize();

	return vec;
} 

UVector3 ShapeTester::GetPointOnOrb(double r) 
{
	double phi = 2.*UUtils::kPi*UUtils::Random();
	double theta = UUtils::ACos(1.-2.*UUtils::Random());
	double vx = std::sin(theta)*std::cos(phi);
	double vy = std::sin(theta)*std::sin(phi);
	double vz = std::cos(theta);
	UVector3 vec(vx,vy,vz);
	vec.Normalize();
        vec=vec*r;
	return vec;
} 

// DONE: all set point methods are performance equivalent


int ShapeTester::TestConsistencySolids()
{
  int errCode= 0;

  std::cout<<"% Performing CONSISTENCY TESTS: ConsistencyTests for Inside, Outside and Surface points " <<std::endl;

  errCode+= TestInsidePoint();
  errCode+= TestOutsidePoint();
  errCode+= TestSurfacePoint();
  
  if (ifSaveAllData){
    UVector3 point;
    for (int i = 0; i < maxPoints; i++)
    {
      GetVectorUSolids(point, points, i);
      VUSolid::EnumInside inside = volumeUSolids->Inside(point);
      resultDoubleUSolids[i] = (double) inside;
    }
    SaveResultsToFile("Inside");
  }
  return errCode;
}

int ShapeTester::ShapeNormal()
{
  int errCode= 0;
  int nError =0;
  ClearErrors();
  int i;
  int numTrials =1000;
#ifdef VECGEOM_ROOT
  //Visualisation
   TPolyMarker3D *pm2 = 0;
    pm2 = new TPolyMarker3D();
    pm2->SetMarkerSize(0.02);
    pm2->SetMarkerColor(kBlue);
#endif
  UVector3 minExtent,maxExtent;
  volumeUSolids->Extent(minExtent,maxExtent);
  double maxX=std::max(std::fabs(maxExtent.x()),std::fabs(minExtent.x()));
  double maxY=std::max(std::fabs(maxExtent.y()),std::fabs(minExtent.y()));
  double maxZ=std::max(std::fabs(maxExtent.z()),std::fabs(minExtent.z()));
  double maxXYZ=2*std::sqrt(maxX*maxX+maxY*maxY+maxZ*maxZ);
  double step = maxXYZ*VUSolid::Tolerance();
  for ( i = 0; i < maxPointsInside; i++)
  {
   UVector3 point = points[i+offsetInside];
   UVector3 dir = directions[i+offsetInside];    
   UVector3 norm;
   bool convex;
      
       VUSolid::EnumInside inside;
       int count = 0;
       double dist=volumeUSolids->DistanceToOut(point, dir ,norm,convex);
       point = point + dist*dir;
       for (int j = 0; j < numTrials; j++)
       { 
         UVector3 dir_new;
         do{
           dir_new=GetRandomDirection();
	   inside = volumeUSolids->Inside(point+dir_new*step);
	   count++;
         }while((inside!=vecgeom::EInside::kInside)&&(count < 1000));
           
         if(count>=1000)
         {ReportError( &nError,point, dir_new, 0, "SN: Can not find direction pointing Inside after 1000 trials");
            break;
         }
	 count = 0;
         dist=volumeUSolids->DistanceToOut(point, dir_new ,norm,convex);
         if ( dist <  VUSolid::Tolerance() ) {
	   if(inside == vecgeom::EInside::kInside)
           ReportError( &nError,point, dir_new, dist, "SN: DistanceToOut has to be  bigger than tolerance for point Inside");
         }
          if ( dist >=  UUtils::kInfinity ) {
	 
           ReportError( &nError,point, dir_new, dist, "SN: DistanceToOut has to be finite number");
         }
         double dot=norm.Dot(dir_new);
         if ( dot < 0. )
         {
           ReportError( &nError,point, dir_new, dot, "SN: Wrong direction of Normal calculated by DistanceToOut");
         }
         if(definedNormal)
         {
           UVector3 normal;
           bool valid =volumeUSolids->Normal(point,normal);
           if ( ! valid ) ReportError( &nError,point, dir_new, 0, "SN: Normal has to be valid for point on the Surface");
           dot = normal.Dot(dir_new);
            if ( dot < 0. )
            {
             ReportError( &nError,point, dir_new, dot, "SN: Wrong direction of Normal calculated by Normal");
             }
     
         }
         point = point + dist*dir_new;
#ifdef VECGEOM_ROOT
         //visualisation
         pm2->SetNextPoint(point.x(),point.y(),point.z());
#endif
         if(volumeUSolids->Inside(point)==vecgeom::EInside::kOutside)
         {
           ReportError( &nError,point, dir_new, 0, "SN: DistanceToOut is overshooting,  new point must be on the Surface"); break;
         }
         double safFromIn = volumeUSolids->SafetyFromInside (point);
         double safFromOut = volumeUSolids->SafetyFromOutside (point);
         if ( safFromIn > VUSolid::Tolerance()) ReportError( &nError,point, dir_new, safFromIn, "SN: SafetyFromInside must be zero on Surface ");
         if ( safFromOut > VUSolid::Tolerance()) ReportError( &nError,point, dir_new, safFromOut, "SN: SafetyFromOutside must be zero on Surface");
        
      }
      
  }   
 
#ifdef VECGEOM_ROOT
    //visualisation
    new TCanvas("shape03", "ShapeNormals", 1000, 800);
    pm2->Draw();
#endif
   std::cout<<"% "<<std::endl;    
   std::cout << "% TestShapeNormal reported = " << CountErrors() << " errors"<<std::endl; 
   std::cout<<"% "<<std::endl; 

   if( CountErrors() )
     errCode= 256; // errCode: 0001 0000 0000

   return errCode;
}

int ShapeTester::ShapeDistances()
{
  int errCode= 0;
  int i;
  int nError = 0 ;
  ClearErrors();
  double maxDifOut=0, maxDifIn=0., delta =0.,tolerance=VUSolid::Tolerance();
  bool convex,convex2;
  UVector3 norm;
  UVector3 minExtent,maxExtent;

  volumeUSolids->Extent(minExtent,maxExtent);
  double maxX=std::max(std::fabs(maxExtent.x()),std::fabs(minExtent.x()));
  double maxY=std::max(std::fabs(maxExtent.y()),std::fabs(minExtent.y()));
  double maxZ=std::max(std::fabs(maxExtent.z()),std::fabs(minExtent.z()));
  double maxXYZ=2*std::sqrt(maxX*maxX+maxY*maxY+maxZ*maxZ);
  double dmove = maxXYZ;

 #ifdef VECGEOM_ROOT 
  //Histograms
   TH1D *hist1 = new TH1D("Residual", "Residual DistancetoIn/Out",200,-20, 0);
     hist1->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
     hist1->GetYaxis()->SetTitle("count");
     hist1->SetMarkerStyle(kFullCircle);
   TH1D *hist2 = new TH1D("AccuracyIn", "Accuracy distanceToIn for points near Surface",200,-20, 0);
     hist2->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
     hist2->GetYaxis()->SetTitle("count");
     hist2->SetMarkerStyle(kFullCircle);
  TH1D *hist3 = new TH1D("AccuracyOut", "Accuracy distanceToOut for points near Surface",200,-20, 0);
     hist3->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
     hist3->GetYaxis()->SetTitle("count");
     hist3->SetMarkerStyle(kFullCircle);
#endif
      
  for ( i = 0; i < maxPointsInside; i++)
  {
   UVector3 point = points[i+offsetInside];
   UVector3 dir = directions[i+offsetInside];    
   double DistanceOut2 = volumeUSolids->DistanceToOut(point, dir ,norm,convex2);
   
   UVector3 pointIn = point+dir*DistanceOut2*(1.-10*tolerance);
   double DistanceOut =  volumeUSolids->DistanceToOut(pointIn, dir ,norm,convex);
   UVector3 pointOut = point+dir*DistanceOut2*(1.+10*tolerance);
   double DistanceIn =  volumeUSolids-> DistanceToIn(pointOut, -dir );
   //Calculate distances for convex or notconvex case 
   double DistanceToInSurf = volumeUSolids->DistanceToIn(point+dir*DistanceOut2,dir);
   if(DistanceToInSurf >= UUtils::kInfinity )
   {
     dmove = maxXYZ;
     if( !convex2 )
     { bool testConvexity = false;
       UVector3 pointSurf = point+dir*DistanceOut2;
       for (int k = 0; k < 100; k++)
       {
         UVector3 rndDir = GetRandomDirection();
         double distTest = volumeUSolids->DistanceToIn(pointSurf,rndDir);
         if((distTest <= UUtils::kInfinity )&&(distTest>0.))
           {testConvexity=true;break;}
       }
       if(!testConvexity)ReportError( &nError,point, dir, DistanceToInSurf, "SD: Error in convexity, must be convex"); 
     }
  
   }
   else
   {//reentering solid, it is not convex
     dmove = DistanceToInSurf*0.5;
     if( convex2 )ReportError( &nError,point, dir, DistanceToInSurf, "SD: Error in convexity, must be NOT convex"); 
   }
    double DistanceToIn2 =  volumeUSolids-> DistanceToIn(point+dir*dmove, -dir );
   //double Dif=maxXYZ-DistanceIn-DistanceOut2;
   //std::cout<<"Diff="<<Dif<<std::endl;
   
   if(DistanceOut > 1000.*tolerance)
   ReportError( &nError,pointIn, dir, DistanceOut, "SD: DistanceToOut is not precise");
   if(DistanceIn > 1000.*tolerance)
   ReportError( &nError,pointOut, dir, DistanceIn, "SD: DistanceToIn is not precise "); 

   if( maxDifOut < DistanceOut ) { maxDifOut = DistanceOut;}
   if( ( volumeUSolids->Inside(pointOut-dir*DistanceIn)!=vecgeom::EInside::kOutside )&&(maxDifIn < DistanceIn ))   
   { maxDifIn = DistanceIn;}
   
    double difDelta = dmove-DistanceOut2 - DistanceToIn2 ;
    if(difDelta > 1000.*tolerance)
    ReportError( &nError,point, dir, difDelta, "SD: Distances calculation is not precise");  
    if ( difDelta > delta) delta=std::fabs (difDelta) ; 
    
#ifdef VECGEOM_ROOT
    //Hstograms
    if(std::fabs(difDelta) < 1E-20) difDelta = 1E-30;
    if(std::fabs(DistanceIn) < 1E-20) difDelta = 1E-30;
    if(std::fabs(DistanceOut) < 1E-20) difDelta = 1E-30;
    hist1->Fill(std::max(0.5*std::log(std::fabs(difDelta)),-20.)); 
    hist2->Fill(std::max(0.5*std::log(std::fabs(DistanceIn)),-20.)); 
    hist3->Fill(std::max(0.5*std::log(std::fabs(DistanceOut)),-20.)); 
#endif
   

  }
  if(fVerbose){
   std::cout<<"% TestShapeDistances:: Accuracy max for DistanceToOut="<<maxDifOut<<" from asked accuracy eps="<<10*tolerance<<std::endl;
   std::cout<<"% TestShapeDistances:: Accuracy max for DistanceToIn="<<maxDifIn<<" from asked accuracy eps="<<10*tolerance<<std::endl;
   std::cout<<"% TestShapeDistances:: Accuracy max for Delta="<<delta<<std::endl;
  }
   std::cout<<"% "<<std::endl; 
   std::cout << "% TestShapeDistances reported = " << CountErrors() << " errors"<<std::endl; 
   std::cout<<"% "<<std::endl; 

   if( CountErrors() )
     errCode= 32; // errCode: 0000 0010 0000

#ifdef VECGEOM_ROOT
   //Histograms
   TCanvas *c4=new TCanvas("c4", "Residuals DistancsToIn/Out", 800, 600);
   c4->Update();
    hist1->Draw();
    TCanvas *c5=new TCanvas("c5", "Residuals DistancsToIn", 800, 600);
   c5->Update();
    hist2->Draw();
 TCanvas *c6=new TCanvas("c6", "Residuals DistancsToOut", 800, 600);
   c6->Update();
    hist3->Draw();
#endif
  
    return errCode;
}

int ShapeTester::TestNormalSolids()
{
  int errCode= 0;
  UVector3 point, normal;

  for (int i = 0; i < maxPoints; i++)
  {
    GetVectorUSolids(point, points, i);
    bool valid = volumeUSolids->Normal(point, normal);
    if (ifSaveAllData) 
    {
      resultBoolUSolids[i] = valid;
      SetVectorUSolids(normal, resultVectorUSolids, i);
    }
  }
  
  SaveResultsToFile("Normal");

  return errCode;
}

int ShapeTester::TestSafetyFromOutsideSolids()
{
  int errCode= 0;
  std::cout<<"% Performing SAFETYFromOUTSIDE TESTS: ShapeSafetyFromOutside " <<std::endl;
  errCode+= ShapeSafetyFromOutside(1000);

  if (ifSaveAllData){
    UVector3 point;
    for (int i = 0; i < maxPoints; i++)
    {
      GetVectorUSolids(point, points, i);
      double res = volumeUSolids->SafetyFromOutside(point,true);
      resultDoubleUSolids[i] = res;
    }
    SaveResultsToFile("SafetyFromOutside");
  }
  
  return errCode;
}

int ShapeTester::TestSafetyFromInsideSolids()
{
  int errCode= 0;
  std::cout<<"% Performing SAFETYFromINSIDE TESTS: ShapeSafetyFromInside " <<std::endl;
  errCode+= ShapeSafetyFromInside(1000);

  if (ifSaveAllData){
    UVector3 point;

    for (int i = 0; i < maxPoints; i++)
    {
      GetVectorUSolids(point, points, i);
      double res = volumeUSolids->SafetyFromInside(point);
      resultDoubleUSolids[i] = res;
    }
  
    SaveResultsToFile("SafetyFromInside");
  }

  return errCode;
}

void ShapeTester::PropagatedNormalU(const UVector3 &point, const UVector3 &direction, double distance, UVector3 &normal)
{
  normal.Set(0);
  if (distance < UUtils::kInfinity)
  {
    UVector3 shift = distance * direction;
    UVector3 surfacePoint = point + shift;
    volumeUSolids->Normal(surfacePoint, normal);
    VUSolid::EnumInside e = volumeUSolids->Inside(surfacePoint);
    if (e != vecgeom::EInside::kSurface)
        e = e;
  }
}

int ShapeTester::TestDistanceToInSolids()
{
  int errCode= 0;
  std::cout<<"% Performing DISTANCEtoIn TESTS: ShapeDistances, TestsAccuracyDistanceToIn and TestFarAwayPoint " <<std::endl;
  errCode+= ShapeDistances();           
  errCode+= TestAccuracyDistanceToIn(1000.);
  errCode+= TestFarAwayPoint();
 
  if (ifSaveAllData) {  
    UVector3 point, direction;
    for (int i = 0; i < maxPoints; i++)
    {
      GetVectorUSolids(point, points, i);
      GetVectorUSolids(direction, directions, i);
      double res = volumeUSolids->DistanceToIn(point, direction);
      resultDoubleUSolids[i] = res;
      
      UVector3 normal;
      PropagatedNormalU(point, direction, res, normal);
      SetVectorUSolids(normal, resultVectorUSolids, i);
      
    }
    SaveResultsToFile("DistanceToIn");
  }

  return errCode;
}

int ShapeTester::TestDistanceToOutSolids()
{
  int errCode= 0;

  std::cout<<"% Performing DISTANCEtoOUT TESTS: Shape Normals " <<std::endl;
  errCode+= ShapeNormal();
  
  if (ifSaveAllData){

    UVector3 point,normal,direction;
    bool convex;
    
    for (int i = 0; i < maxPoints; i++)
    {
      GetVectorUSolids(point, points, i);
      GetVectorUSolids(direction, directions, i);
      normal.Set(0);
      double res = volumeUSolids->DistanceToOut(point, direction, normal, convex);
      
      resultDoubleUSolids[i] = res;
      resultBoolUSolids[i] = convex;
      SetVectorUSolids(normal, resultVectorUSolids, i);
    }
  }
  SaveResultsToFile("DistanceToOut");

  return errCode;
}

int ShapeTester::TestFarAwayPoint()
{
  int errCode= 0;
  UVector3 point,point1,vec, direction, normal,pointSurf;
  int icount=0, icount1=0, nError = 0;
  double distIn,diff, difMax=0., maxDistIn =0.;
  double tolerance = VUSolid::Tolerance();
   ClearErrors();
   
   //for ( int j=0; j<maxPointsSurface+maxPointsEdge; j++)
   for ( int j=0; j<maxPointsInside; j++)
   {
     //point = points[j+offsetSurface];
    point = points[j+offsetInside];
    vec = GetRandomDirection();
    if(volumeUSolids->DistanceToIn(point,vec) < UUtils::kInfinity)continue;
    point1= point;
   
    for (int i=0; i<10000; i++)
    {
          point1 = point1+vec*10000;
    }
    distIn =  volumeUSolids-> DistanceToIn(point1,-vec);
    pointSurf = point1-distIn*vec;
    if( (distIn < UUtils::kInfinity) && (distIn > maxDistIn )) maxDistIn = distIn;

    diff = std::fabs ( (point1 - pointSurf). Mag() - distIn );
    if( diff > 100*tolerance ) icount++;
    if( diff >=  UUtils::kInfinity)
    {icount1++;
       UVector3 temp=-vec;
       ReportError( &nError,point1, temp, diff, "TFA:  Point missed Solid (DistanceToIn = Infinity)");
    }
    else{ if ( diff > difMax ) difMax = diff; }
  }
   if(fVerbose)
   {
   std::cout<<"% TestFarAwayPoints:: number of points with big difference (( DistanceToIn- Dist) ) >  tolerance ="<<icount<<std::endl;
   std::cout <<"%  Maxdif = "<<difMax<<" from MaxDist="<<maxDistIn<<" Number of points missing Solid (DistanceToIn = Infinity) = "<<icount1<<std::endl;
   }
   std::cout<<"% "<<std::endl; 
   std::cout << "% TestFarAwayPoints reported = " << CountErrors() << " errors"<<std::endl;
   std::cout<<"% "<<std::endl; 

   if( CountErrors() )
     errCode= 128; // errCode: 0000 1000 0000

   return errCode;
}

int ShapeTester::TestSurfacePoint()
{
  int errCode= 0;
  UVector3 point, pointSurf,vec,direction, normal;
  bool convex;
  int icount=0, icount1=0;
  double distIn,distOut;
  int iIn=0,iInNoSurf=0,iOut=0,iOutNoSurf=0;
  double tolerance = VUSolid::Tolerance();
  int nError=0;
  ClearErrors();
#ifdef VECGEOM_ROOT
  //Visualisation
   TPolyMarker3D *pm5 = 0;
    pm5 = new TPolyMarker3D();
    pm5->SetMarkerStyle(20);
    pm5->SetMarkerSize(1);
    pm5->SetMarkerColor(kRed);
#endif
 
   for (int i = 0; i < maxPointsSurface+maxPointsEdge; i++)
     { //test GetPointOnSurface()
       point = points[offsetSurface+i];
       #ifdef VECGEOM_ROOT
         //visualisation
         pm5->SetNextPoint(point.x(),point.y(),point.z());
       #endif
       if(volumeUSolids->Inside(point) !=  vecgeom::EInside::kSurface)
       {icount++;   
	 UVector3 v(0,0,0);
         ReportError( &nError,point, v, 0, "TS:  Point on not on the Surface");}
         //test if for point on Surface distIn=distOut=0  
         UVector3 v = GetRandomDirection();
          distIn  = volumeUSolids->DistanceToIn(point,v);
          distOut = volumeUSolids->DistanceToOut(point,v, normal, convex);

          if( distIn == 0. && distOut == 0. )
	    { icount1++;
	      ReportError( &nError,point, v, 0, "TS: DistanceToIn=DistanceToOut=0 for point on Surface");
	  }
        //test Accuracy distance for points near Surface
        pointSurf=point+v*10*tolerance;
        VUSolid::EnumInside inside=volumeUSolids->Inside(pointSurf);
        if(inside != vecgeom::EInside::kSurface)
        {
           if(inside == vecgeom::EInside::kOutside)
           {
            for(int j = 0; j < 1000; j++ )
            {
             vec = GetRandomDirection();
             distIn   = volumeUSolids->DistanceToIn(pointSurf,vec);
	     if(distIn < UUtils::kInfinity)
	    {
              iIn++;
        
              VUSolid::EnumInside surfaceP = volumeUSolids->Inside(pointSurf + distIn*vec);
              if(surfaceP != vecgeom::EInside::kSurface )
	      {
                iInNoSurf++;
	        ReportError( &nError,pointSurf, vec, distIn, "TS: Wrong DistanceToIn for point near Surface");
		        
	      }
            }
	  }
        }
        else
        {
          for(int j = 0; j < 1000; j++ )
          {
            iOut++;
            vec = GetRandomDirection();
            distOut  = volumeUSolids->DistanceToOut(pointSurf, vec,normal, convex); 
            VUSolid::EnumInside surfaceP = volumeUSolids->Inside(pointSurf + distOut*vec);

            if(surfaceP != vecgeom::EInside::kSurface )
	    {
              iOutNoSurf++;
	      ReportError( &nError, pointSurf, vec, distOut, "TS: Wrong DistanceToOut for point near Surface" );
	    }
          }
        }
      

    }
 
  }
   if(fVerbose){
     std::cout<<"% TestSurfacePoints GetPointOnSurface() for Solid  "<<volumeUSolids->GetName()<<" had "<<icount<<" errors"<<std::endl;
     std::cout<<"% TestSurfacePoints both  DistanceToIN and DistanceToOut ==0 for "<<volumeUSolids->GetName()<<" had "<<icount1<<" errors"<<std::endl;
     std::cout<<"% TestSurfacePoints new moved point is not on Surface::iInNoSurf = "<<iInNoSurf<<";    iOutNoSurf = "<<iOutNoSurf<<std::endl;
  
   }
   #ifdef VECGEOM_ROOT
    //visualisation
    new TCanvas("shape05", "GetPointOnSurface", 1000, 800);
    pm5->Draw();
    #endif
   std::cout<<"% "<<std::endl; 
   std::cout << "% Test Surface Point reported = " << CountErrors() << " errors"<<std::endl;	
   std::cout<<"% "<<std::endl; 

   if( CountErrors() )
     errCode= 4; // errCode: 0000 0000 0100

   return errCode;
}

int ShapeTester::TestInsidePoint()
{
  int errCode= 0;
  int i, n = maxPointsOutside;
  int nError = 0;
  ClearErrors();
 
  UVector3 minExtent,maxExtent;
  volumeUSolids->Extent(minExtent,maxExtent);
  double maxX=std::max(std::fabs(maxExtent.x()),std::fabs(minExtent.x()));
  double maxY=std::max(std::fabs(maxExtent.y()),std::fabs(minExtent.y()));
  double maxZ=std::max(std::fabs(maxExtent.z()),std::fabs(minExtent.z()));
  double maxXYZ=2*std::sqrt(maxX*maxX+maxY*maxY+maxZ*maxZ);

 for (int j = 0; j < maxPointsInside; j++)
 {
    //Check values of Safety
    UVector3 point=points[j+offsetInside]; 
    double safeDistance = volumeUSolids->SafetyFromInside(point );
    if (safeDistance <= 0.0) {
	UVector3 zero(0);
	ReportError( &nError, point, zero, safeDistance, "TI: SafetyFromInside(p) <= 0");

	if( CountErrors() )
	  errCode= 1; // errCode: 0000 0000 0001
	
	return errCode;
    }
    double safeDistanceFromOut = volumeUSolids->SafetyFromOutside(point );
    if (safeDistanceFromOut > 0.0) {
	UVector3 zero(0);
	ReportError(  &nError, point, zero, safeDistanceFromOut, "TI: SafetyFromOutside(p) not 0 for Point Inside" );
	//continue;
    }
   
    //Check values of Extent
    
    if (point.x() < minExtent.x() ||
	point.x() > maxExtent.x() || 
        point.y() < minExtent.y() || 
        point.y() > maxExtent.y() || 
        point.z() < minExtent.z() || 
	point.z() > maxExtent.z() ) {
	 UVector3 zero(0);
         ReportError(  &nError, point, zero, safeDistance, "TI: Point is outside Extent");
          } 

     //Check values with points and directions to outside points
     for( i=0; i < n; i++ ) 
     {
       UVector3 vr = points[i+offsetOutside] - point;
       UVector3 v = vr.Unit();
       bool valid,convex;
       valid=false;
       UVector3 norm;

       double dist = volumeUSolids->DistanceToOut( point, v, norm,convex);
       double NormalDist ;

       NormalDist = volumeUSolids->SafetyFromInside( point );
     
      if (dist > maxXYZ) {
        ReportError( &nError, point, v, dist, "TI: DistanceToOut(p,v) > Solid's Extent  dist = ");
      continue;
      }
      if (dist <= 0) {
        ReportError( &nError, point, v, NormalDist, "TI: DistanceToOut(p,v) <= 0  Normal Dist = ");
      continue;
      }
      if (dist >= UUtils::kInfinity) {
        ReportError( &nError, point, v, safeDistance, "TI: DistanceToOut(p,v) == kInfinity" );
      continue;
      }
      if (dist < safeDistance-1E-10) {
       ReportError( &nError, point, v, safeDistance, "TI: DistanceToOut(p,v) < DistanceToIn(p)");
      continue;
      }

      if (valid) {
      if (norm.Dot(v) < 0) {
         ReportError( &nError, point, v, safeDistance, "TI: Outgoing normal incorrect" );
	continue;
       }
      }
      //Check DistanceToIn, 0 for now, has to be -1 in future
      double distIn = volumeUSolids->DistanceToIn( point, v);
      if (distIn > 0.) {
	//ReportError( nError, point, v, distIn, "TI: DistanceToIn(p,v) has to be 0 or negative");
	//std::cout<<"distIn="<<distIn<<std::endl;
      continue;
      }
      //Move to the boundary and check
       UVector3 p = point + v*dist;
    
       VUSolid::EnumInside insideOrNot = volumeUSolids->Inside(p);
       if (insideOrNot == vecgeom::EInside::kInside) {
        ReportError( &nError, point, v, safeDistance, "TI: DistanceToOut(p,v) undershoots" );
      continue;
       }
       if (insideOrNot == vecgeom::EInside::kOutside) {
        ReportError( &nError, point, v, safeDistance, "TI: DistanceToOut(p,v) overshoots" );
       continue;
       }
       UVector3 norm1;
        valid = volumeUSolids->Normal( p ,norm1);
        if (norm1.Dot(v) < 0) {
	  if (volumeUSolids->DistanceToIn(p,v) != 0){
           ReportError( &nError, p, v, safeDistance, "TI: SurfaceNormal is incorrect" );
	  }
    }//End Check points and directions
  }
 }
    std::cout<<"% "<<std::endl; 
    std::cout << "% TestInsidePoint reported = " << CountErrors() << " errors"<<std::endl;
    std::cout<<"% "<<std::endl; 

    
    if( CountErrors() )
      errCode= 1; // errCode: 0000 0000 0001

    return errCode;
}

int ShapeTester::TestOutsidePoint( )
{
  int errCode= 0;
  int i, n = maxPointsInside;
  int nError=0;
  ClearErrors();

  for( int j=0; j < maxPointsOutside; j++ ) {
    //std::cout<<"ConsistencyOutside check"<<j<<std::endl;
  UVector3 point = points[j+offsetOutside];
  double safeDistance = volumeUSolids->SafetyFromOutside( point );
  
  if (safeDistance <= 0.0) {
    UVector3 zero(0);
    ReportError(  &nError, point, zero, safeDistance,"T0: SafetyFromOutside(p) <= 0");

    if( CountErrors() )
      errCode= 2; // errCode: 0000 0000 0010
    
    return errCode;
  }

   double safeDistanceFromInside = volumeUSolids->SafetyFromInside( point );
  
  if (safeDistanceFromInside > 0.0) {
	UVector3 zero(0);
    ReportError(  &nError, point, zero, safeDistanceFromInside,"T0: SafetyFromInside(p) not 0 for point Outside");
    //continue;
  }
   
   for( i=0; i < n; i++ ) {
    UVector3 vr = points[i+offsetInside] - point;
    UVector3 v = vr.Unit();

    double dist = volumeUSolids->DistanceToIn( point, v );
    if (dist <= 0) {
      ReportError(  &nError, point, v, safeDistance, "T0: DistanceToIn(p,v) <= 0" );
      continue;
    }
    if (dist >= UUtils::kInfinity) {
      ReportError(&nError, point, v, safeDistance, "T0: DistanceToIn(p,v) == kInfinity" );
      continue;
    }
    if (dist < safeDistance-1E-10) {
      ReportError(  &nError, point, v, safeDistance, "T0: DistanceToIn(p,v) < DistanceToIn(p)" );
      continue;
    }

    UVector3 p = point + dist*v;
    VUSolid::EnumInside insideOrNot = volumeUSolids->Inside( p );
    if (insideOrNot == vecgeom::EInside::kOutside) {
      ReportError(  &nError, point, v, safeDistance, "T0: DistanceToIn(p,v) undershoots");
      continue;
    }
    if (insideOrNot == vecgeom::EInside::kInside) {
      ReportError(  &nError, point, v, safeDistance, "TO: DistanceToIn(p,v) overshoots" );
      continue;
    }

    dist = volumeUSolids->SafetyFromOutside( p );

    //if (dist != 0) {
    if (dist > VUSolid::Tolerance()) {
      ReportError(  &nError, p, v, safeDistance, "T02: DistanceToIn(p) should be zero" );
      // logger << "Dist != 0 : " << dist << endl;
      continue;
    }

    dist = volumeUSolids->SafetyFromInside( p );
    //if (dist != 0) {
    if (dist > VUSolid::Tolerance()) {
      ReportError(&nError , p, v, safeDistance, "T02: DistanceToOut(p) should be zero" );
      continue;
    }

    dist = volumeUSolids->DistanceToIn( p, v );
    safeDistance = volumeUSolids->SafetyFromOutside( p );
    //
    // Beware! We might expect dist to be precisely zero, but this may not
    // be true at corners due to roundoff of the calculation of p = point + dist*v.
    // It should, however, *not* be infinity.
    //
    //if (dist != UUtils::kInfinity) {
     if (dist >= UUtils::kInfinity) {
      ReportError(  &nError, p, v, safeDistance, "T02: DistanceToIn(p,v) == kInfinity" );
      continue;
    }	

    bool valid,convex,convex1;
    valid=false;
    UVector3 norm;

    dist = volumeUSolids->DistanceToOut( p, v, norm, convex );
    if (dist == 0) continue;

    if (dist >= UUtils::kInfinity) {
      ReportError(  &nError, p, v, safeDistance, "T02: DistanceToOut(p,v) == kInfinity" );
      continue;
    }
    else if (dist < 0) {
      ReportError(  &nError, p, v, safeDistance, "T02: DistanceToOut(p,v) < 0");
      continue;
    }

    if (valid) {
      if (norm.Dot(v) < 0) {
	ReportError(  &nError, p, v, safeDistance, "T02: Outgoing normal incorrect" );
	continue;
      }
    }

    UVector3 norm1;
    valid = volumeUSolids->Normal( p,norm1 );
    if (norm1.Dot(v) > 0) {
      ReportError(  &nError, p, v, safeDistance, "T02: Ingoing surfaceNormal is incorrect" );
    }


    UVector3 p2 = p + v*dist;

    insideOrNot = volumeUSolids->Inside(p2);
    if (insideOrNot == vecgeom::EInside::kInside) {
      ReportError(  &nError, p, v, safeDistance, "T02: DistanceToOut(p,v) undershoots" );
      continue;
    }
    if (insideOrNot == vecgeom::EInside::kOutside) {
      ReportError(  &nError, p, v, safeDistance, "TO2: DistanceToOut(p,v) overshoots" );
      continue;
    }

    UVector3 norm2, norm3 ;
      valid = volumeUSolids->Normal( p2 , norm2);
    if (norm2.Dot(v) < 0) {
      if (volumeUSolids->DistanceToIn(p2,v) != 0)
	ReportError(  &nError, p2, v, safeDistance, "T02: Outgoing surfaceNormal is incorrect" );
    }
    if (convex) {
      if (norm.Dot(norm2) < 0.0) {
	ReportError(  &nError, p2, v, safeDistance, "T02: SurfaceNormal and DistanceToOut disagree on normal" );
      }
    }

    if (convex) {
      dist = volumeUSolids->DistanceToIn(p2,v);
      if (dist == 0) {
	//
	// We may have grazed a corner, which is a problem of design.
	// Check distance out
	//
	if (volumeUSolids->DistanceToOut(p2,v,norm3,convex1) != 0) {
	  ReportError(  &nError, p, v, safeDistance, "TO2: DistanceToOut incorrectly returns validNorm==true (line of sight)(c)");
	  continue;
	}
      }
      else if (dist != UUtils::kInfinity) {
	//ReportError(  &nError, p, v, safeDistance, "TO2: DistanceToOut incorrectly returns validNorm==true (line of sight)" );
	continue;
      }

      int k;
      //for( k=0; k < n; k++ ) {
        for( k=0; k < 10; k++ ) {
	UVector3 p2top = points[k+offsetInside] - p2;

	if (p2top.Dot(norm) > 0) {
	  ReportError(  &nError, p, v,safeDistance, 
		       "T02: DistanceToOut incorrectly returns validNorm==true (horizon)" );
	  continue;
	}
      }
    } // if valid normal
  } // Loop over inside points
 
   n = maxPointsOutside;

  for(int l=0; l < n; l++ ) {
    UVector3 vr =  points[l+offsetOutside] - point;
    if (vr.Mag2() < DBL_MIN) continue;

    UVector3 v = vr.Unit();

    double dist = volumeUSolids->DistanceToIn( point, v );

    if (dist <= 0) {
      ReportError(  &nError, point, v, safeDistance, "T03: DistanceToIn(p,v) <= 0" );
      continue;
    }
    if (dist >= UUtils::kInfinity) {
      //G4cout << "dist == kInfinity" << G4endl ;
      continue;
    }
    if (dist < safeDistance-1E-10) {
      ReportError(  &nError, point, v, safeDistance, "T03: DistanceToIn(p,v) < DistanceToIn(p)" );
      continue;
    }
    UVector3 p = point + dist*v;

     VUSolid::EnumInside insideOrNot = volumeUSolids->Inside( p );
     if (insideOrNot == vecgeom::EInside::kOutside) {
      ReportError(  &nError, point, v, safeDistance, "T03: DistanceToIn(p,v) undershoots" );
      continue;
    }
     if (insideOrNot == vecgeom::EInside::kInside) {
      ReportError(  &nError, point, v, safeDistance, "TO3: DistanceToIn(p,v) overshoots");
      continue;
    }
  } // Loop over outside points

  }
   std::cout<<"% "<<std::endl; 
   std::cout<< "% TestOutsidePoint reported = " << CountErrors() << " errors"<<std::endl;
   std::cout<<"% "<<std::endl; 

   if( CountErrors() )
     errCode= 2; // errCode: 0000 0000 0010

   return errCode;
}
//
//Surface Checker 
//
int ShapeTester::TestAccuracyDistanceToIn(double dist)
{
  int errCode= 0;
  UVector3 point,pointSurf,pointIn,v, direction, normal;
  bool convex;
  double distIn,distOut;
  double maxDistIn=0,diff=0,difMax=0;
  int nError=0;
  ClearErrors();
  int  iIn=0,iInNoSurf=0,iOut=0,iOutNoSurf=0,iWrongSideIn=0,iWrongSideOut=0,
        iOutInf=0,iOutZero=0,iInInf=0,iInZero=0,iSafIn=0,iSafOut=0;
  double tolerance = VUSolid::Tolerance();

#ifdef VECGEOM_ROOT
  //Histograms
   TH1D *hist10 = new TH1D("AccuracySurf", "Accuracy DistancetoIn",200,-20, 0);
     hist10->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
     hist10->GetYaxis()->SetTitle("count");
     hist10->SetMarkerStyle(kFullCircle);
#endif

 //test Accuracy distance 
     for (int i = 0; i < maxPointsSurface+maxPointsEdge; i++)
     { 
      
      //test GetPointOnSurface
      pointSurf = points[i+offsetSurface];
      UVector3 vec = GetRandomDirection();

      point=pointSurf+vec*dist; 

      VUSolid::EnumInside inside=volumeUSolids->Inside(point);
     
      if(inside !=  vecgeom::EInside::kSurface)
      {
           if(inside ==  vecgeom::EInside::kOutside)
           {
            distIn   = volumeUSolids->DistanceToIn(pointSurf,vec);
            if(distIn >= UUtils::kInfinity){ 
             // Accuracy Test for convex part 
              distIn   = volumeUSolids->DistanceToIn(point,-vec);
              if(maxDistIn < distIn)maxDistIn = distIn;
              diff = ( (pointSurf-point).Mag() - distIn);
              if(diff > difMax) difMax = diff;
               if(std::fabs(diff) < 1E-20) diff = 1E-30;
#ifdef VECGEOM_ROOT
               hist10->Fill(std::max(0.5*std::log(std::fabs(diff)),-20.)); 
#endif
	    }
            
            // Test for consistency for points situated Outside 
            for(int j = 0; j < 1000; j++ )
            {
             vec = GetRandomDirection();
             
             distIn   = volumeUSolids->DistanceToIn(point,vec);
             distOut = volumeUSolids->DistanceToOut(point,vec, normal, convex);
             iWrongSideOut++;
             if(distOut>=UUtils::kInfinity){
              iOutInf++;
              ReportError(  &nError, point, vec, distOut, "TAD: DistanceToOut is Infinity for point Outside");
             }
             if(std::fabs(distOut) < -tolerance){
              iOutZero++;
              ReportError(  &nError, point, vec, distOut, "TAD: DistanceToOut is < tolerance for point Outside");
             }
             double safFromIn = volumeUSolids->SafetyFromInside(point);
             if(safFromIn > tolerance){
	      iSafOut++;
              ReportError(  &nError, point, vec,safFromIn , "TAD: SafetyFromInside is > tolerance for point Outside");
	     }

             // Test for consistency for points situated Inside
             pointIn = pointSurf +vec*1000.*VUSolid::Tolerance();
             if(volumeUSolids->Inside(pointIn) ==  vecgeom::EInside::kInside)
	     {
               double distOut1  = volumeUSolids->DistanceToOut(pointIn,vec, normal, convex); 
	       VUSolid::EnumInside surfaceP = volumeUSolids->Inside(pointIn + distOut1*vec);
               double distIn1   = volumeUSolids->DistanceToIn(pointIn,vec);
	       iWrongSideIn++;
               if(distOut1>=UUtils::kInfinity){
                iInInf++;
                ReportError(  &nError, pointIn, vec,distOut1 , "TAD: Distance ToOut is Infinity  for point Inside");
               }
               if(std::fabs(distOut1)<tolerance){
                iInZero++;
                ReportError(  &nError, pointIn, vec,distOut1 , "TAD: Distance ToOut < tolerance  for point Inside");
               }
               if(std::fabs(distIn1)>tolerance){
                iInZero++;
                ReportError(  &nError, pointIn, vec,distIn1 , "TAD: Distance ToIn > tolerance  for point Inside");
               }
               double safFromOut = volumeUSolids->SafetyFromOutside(pointIn);
               if(safFromOut > tolerance){
                iSafIn++;
                ReportError(  &nError, pointIn, vec,safFromOut , "TAD: SafetyFromOutside > tolerance  for point Inside");
	      }
	      iIn++;
              if(surfaceP != vecgeom::EInside::kSurface )
	      {
               iOutNoSurf++;
	       ReportError(  &nError, pointIn, vec, 0. , "TAD: Moved to Surface point is not on Surface");
	      }
	     }

	     // Test for consistency for points situated on Surface
	    if(distIn <  UUtils::kInfinity)
	    {
              iIn++;
              
	      //Surface Test 
	      VUSolid::EnumInside surfaceP = volumeUSolids->Inside(point + distIn*vec);
              if(surfaceP !=  vecgeom::EInside::kSurface )
	      {
                iInNoSurf++;
	        ReportError(  &nError, point, vec, 0. , "TAD: Moved to Solid point is not on Surface");
	      }
            }
	  }
        }
        else
        {
          for(int j = 0; j < 1000; j++ )
          {
            iOut++;
            vec = GetRandomDirection();
                   
            distOut  = volumeUSolids->DistanceToOut(point,vec, normal, convex); 
	    VUSolid::EnumInside surfaceP = volumeUSolids->Inside(point + distOut*vec);
             distIn   = volumeUSolids->DistanceToIn(point,vec);
	     iWrongSideIn++;
             if(distOut>=UUtils::kInfinity){
              iInInf++;
              ReportError(  &nError, point, vec,distOut , "TAD: Distance ToOut is Infinity  for point Inside");
             }
             if(std::fabs(distOut)<tolerance){
              iInZero++;
              ReportError(  &nError, point, vec,distOut , "TAD: Distance ToOut < tolerance  for point Inside");
             }
             if(std::fabs(distIn)>tolerance){
              iInZero++;
              ReportError(  &nError, point, vec,distOut , "TAD: Distance ToIn > tolerance  for point Inside");
             }
             double safFromOut = volumeUSolids->SafetyFromOutside(point);
             if(safFromOut > tolerance){
              iSafIn++;
              ReportError(  &nError, point, vec,safFromOut , "TAD: SafetyFromOutside > tolerance  for point Inside");
	     }
             if(surfaceP != vecgeom::EInside::kSurface )
	     {
              iOutNoSurf++;
	      ReportError(  &nError, point, vec, 0. , "TAD: Moved to Surface point is not on Surface");
	    }
          }
        }
      
     }
    }
   if(fVerbose){
     // Surface
     std::cout<<"TestAccuracyDistanceToIn::Errors for moved point is not on Surface ::iInNoSurf = "<<iInNoSurf<<";    iOutNoSurf = "<<iOutNoSurf<<std::endl;
     std::cout<<"TestAccuracyDistanceToIn::Errors SolidUSolid ::From total number of points  = "<<iIn<<std::endl;
     // Distances
     std::cout<<"TestForWrongSide Error in DistanceToOut:: Point is Outside DistanceToOut not zero = " << iOutZero <<" from total number "<<iWrongSideOut <<"\n";
     std::cout<<"TestForWrongSide Error in DistanceToOut:: Point is Outside DistanceToOut is Infinity = " << iOutInf <<" from total number "<<iWrongSideOut <<"\n";
     std::cout<<"TestForWrongSide Error in DistanceToIn:: Point is Inside DistanceToIn not zero = " << iInZero<<" from total number "<<iWrongSideIn<< "\n";
     std::cout<<"TestForWrongSide Error in DistanceToIn:: Point is Inside DistanceToIn not zero = " << iInInf<<" from total number "<<iWrongSideIn<< "\n";
     // Safety
     std::cout<< "TestForWrongSide Error in Safety:: Point is Outside SafetyFromInside not zero = " << iSafOut << "\n";
     std::cout<< "TestForWrongSide Error in Safety:: Point is Inside SafetyFromOutside not zero = " << iSafIn << "\n";
   
   }
#ifdef VECGEOM_ROOT
    TCanvas *c7=new TCanvas("c7", "Accuracy DistancsToIn", 800, 600);
    c7->Update();
    hist10->Draw();
#endif
    std::cout<<"% "<<std::endl; 
    std::cout << "% TestAccuracyDistanceToIn reported = " << CountErrors() << " errors"<<std::endl;
    std::cout<<"% "<<std::endl; 

    if( CountErrors() )
      errCode= 64; // errCode: 0000 0100 0000

    return errCode;
}

int  ShapeTester::ShapeSafetyFromInside(int max)
{
  int errCode= 0;
  UVector3 point,dir,pointSphere,norm;
  bool convex;
  int count=0, count1=0;
  int nError =0;
  ClearErrors();
#ifdef VECGEOM_ROOT
  //visualisation
   TPolyMarker3D *pm3 = 0;
    pm3 = new TPolyMarker3D();
    pm3->SetMarkerSize(0.2);
    pm3->SetMarkerColor(kBlue);
#endif

  if( max > maxPoints )max=maxPoints;
  for (int i = 0; i < max; i++)
  {
   GetVectorUSolids(point, points, i);
   double res = volumeUSolids->SafetyFromInside(point);
   for (int j=0;j<1000;j++)
   { 
     dir=GetRandomDirection();
     pointSphere=point+res*dir;
#ifdef VECGEOM_ROOT
     //visualisation
     pm3->SetNextPoint(pointSphere.x(),pointSphere.y(),pointSphere.z());
#endif
     double distOut=volumeUSolids->DistanceToOut(point,dir,norm,convex);
     if(distOut < res) {count1++;
        ReportError(  &nError, pointSphere, dir, distOut, "SSFI: DistanceToOut is underestimated,  less that Safety" );
     }
     if( volumeUSolids->Inside(pointSphere) == vecgeom::EInside::kOutside)
     { 
       ReportError(  &nError, pointSphere, dir, res, "SSFI: Safety is not safe, point on the SafetySphere is Outside" );
       double error=volumeUSolids->DistanceToIn(pointSphere,-dir);
       if(error>100*VUSolid::Tolerance())
       {   
	count++;
        
       }   
      }
     }
   }
   if(fVerbose){
     std::cout<<"% "<<std::endl; 
     std::cout<<"% ShapeSafetyFromInside ::  number of Points Outside Safety="<<count<<" number of points with  distance smaller that safety="<<count1<<std::endl;
     std::cout<<"% "<<std::endl; 
   }
#ifdef VECGEOM_ROOT
     //visualisation
    new TCanvas("shape", "ShapeSafetyFromInside", 1000, 800);
    pm3->Draw();
#endif
    std::cout<<"% "<<std::endl; 
    std::cout<< "% TestShapeSafetyFromInside reported = " << CountErrors() << " errors"<<std::endl;
    std::cout<<"% "<<std::endl; 

    if( CountErrors() )
      errCode= 8; // errCode: 0000 0000 1000

    return errCode;
}

int ShapeTester::ShapeSafetyFromOutside(int max)
{
  int errCode= 0;
  UVector3 point,temp,dir,pointSphere,normal;
  double res,error;
  int count=0, count1=0;
  int nError;
  ClearErrors();
#ifdef VECGEOM_ROOT
  //visualisation
   TPolyMarker3D *pm4 = 0;
    pm4 = new TPolyMarker3D();
    pm4->SetMarkerSize(0.2);
    pm4->SetMarkerColor(kBlue);
#endif

  UVector3 minExtent,maxExtent;
  volumeUSolids->Extent(minExtent,maxExtent);
  //double maxX=std::max(std::fabs(maxExtent.x()),std::fabs(minExtent.x()));
  //double maxY=std::max(std::fabs(maxExtent.y()),std::fabs(minExtent.y()));
  //double maxZ=std::max(std::fabs(maxExtent.z()),std::fabs(minExtent.z()));
  //double maxXYZ= std::sqrt(maxX*maxX+maxY*maxY+maxZ*maxZ);
  if( max > maxPointsOutside )max=maxPointsOutside;
  for (int i = 0; i < max; i++)
  {
    //GetVectorUSolids(point, points, i);
    point=points[i+offsetOutside];
    res = volumeUSolids->SafetyFromOutside(point);
    if(res>0)
    {     //Safety Sphere test
     bool convex;
     int numTrials = 1000;
     //if(res > maxXYZ) 
     //{
     // int dummy = (int)(std::pow((maxXYZ/res),2));
     // numTrials = numTrials*dummy;
     //}
     for (int j=0;j<numTrials;j++)
     { dir=GetRandomDirection();
       double distIn=volumeUSolids->DistanceToIn(point,dir);
       if(distIn < res){count1++;
        ReportError(  &nError, point, dir, distIn, "SSFO: DistanceToIn is underestimated,  less that Safety" );
       }
       pointSphere=point+res*dir;
       //std::cout<<"SFO "<<pointSphere<<std::endl;
#ifdef VECGEOM_ROOT
       //visualisation
       pm4->SetNextPoint(pointSphere.x(),pointSphere.y(),pointSphere.z());
#endif
       if( volumeUSolids->Inside(pointSphere) == vecgeom::EInside::kInside)
       { 
             ReportError(  &nError, pointSphere, dir, res, "SSFO: Safety is not safe, point on the SafetySphere is Inside" );
	error=volumeUSolids->DistanceToOut(pointSphere,-dir,normal,convex);
        if(error>100*VUSolid::Tolerance())
        {   
         count++;
	 
        }   
       }
      }
	  
    }                
 }
 if(fVerbose){
  std::cout<<"% "<<std::endl; 
  std::cout<<"% TestShapeSafetyFromOutside::  number of points Inside Safety Sphere ="<<count<<" number of points with Distance smaller that Safety="<<count1<<std::endl;
  std::cout<<"% "<<std::endl; 
 }
#ifdef VECGEOM_ROOT
 //visualisation
 new TCanvas("shapeTest", "ShapeSafetyFromOutside", 1000, 800);
 pm4->Draw();
#endif
 std::cout<<"% "<<std::endl; 
 std::cout<< "% TestShapeSafetyFromOutside reported = " << CountErrors() << " errors"<<std::endl;
 std::cout<<"% "<<std::endl; 
 
 if( CountErrors() )
   errCode= 16; // errCode: 0000 0001 0000
 
 return errCode;
}
/////////////////////////////////////////////////////////////////////////////
int ShapeTester::TestXRayProfile()
{
  int errCode= 0;

  std::cout<<"% Performing XRayPROFILE number of scans ="<<gNumberOfScans<<std::endl;
  std::cout<<"% \n"<<std::endl; 
  if(gNumberOfScans==1) {errCode+= Integration(0,45,200,true);}//1-theta,2-phi
  else{  errCode+= XRayProfile(0,gNumberOfScans,1000);}
  
  return errCode;
}
/////////////////////////////////////////////////////////////////////////////
int ShapeTester::XRayProfile(double theta, int nphi, int ngrid, bool useeps)
{
  int errCode= 0;

#ifdef VECGEOM_ROOT
  int nError=0; 
  ClearErrors();
  
  TH1F *hxprofile = new TH1F("xprof", Form("X-ray capacity profile of shape %s for theta=%g degrees", volumeUSolids->GetName().c_str(), theta),
			     nphi, 0, 360);
  new TCanvas("c8", "X-ray capacity profile");
  double dphi = 360./nphi;
  double phi = 0;
  double phi0 = 5;
  double maxerr = 0;
  
  for (int i=0; i<nphi; i++) {
    phi = phi0 + (i+0.5)*dphi;
    //graphic option 
    if(nphi==1) {Integration( theta, phi, ngrid,useeps);}
    else{Integration( theta, phi, ngrid,useeps,1,false);}
    hxprofile->SetBinContent(i+1, gCapacitySampled);
    hxprofile->SetBinError(i+1, gCapacityError);
    if (gCapacityError>maxerr) maxerr = gCapacityError;
    if((gCapacitySampled-gCapacityAnalytical)>10*gCapacityError) nError++;
  }
  
  double minval = hxprofile->GetBinContent(hxprofile->GetMinimumBin()) - 2*maxerr;
  double maxval = hxprofile->GetBinContent(hxprofile->GetMaximumBin()) + 2*maxerr;
  hxprofile->GetXaxis()->SetTitle("phi [deg]");
  hxprofile->GetYaxis()->SetTitle("Sampled capacity");
  hxprofile->GetYaxis()->SetRangeUser(minval,maxval);
  hxprofile->SetMarkerStyle(4);
  hxprofile->SetStats(kFALSE);
  hxprofile->Draw();
  TF1 *lin = new TF1("linear",Form("%f",gCapacityAnalytical),0,360);
  lin->SetLineColor(kRed);
  lin->SetLineStyle(kDotted);
  lin->Draw("SAME");
  
  std::cout<<"% "<<std::endl; 
  std::cout<< "% TestShapeRayProfile reported = " << nError << " errors"<<std::endl;
  std::cout<<"% "<<std::endl;  

  if( nError )
    errCode= 1024; // errCode: 0100 0000 0000
#endif

  return errCode;
}
/////////////////////////////////////////////////////////////////////////////
int ShapeTester::Integration(double theta, double phi, int ngrid, bool useeps, int npercell, bool graphics)
{
// integrate shape capacity by sampling rays
  int errCode= 0;
  int nError=0;
  UVector3 minExtent,maxExtent;
  volumeUSolids->Extent(minExtent,maxExtent);
  double maxX=2*std::max(std::fabs(maxExtent.x()),std::fabs(minExtent.x()));
  double maxY=2*std::max(std::fabs(maxExtent.y()),std::fabs(minExtent.y()));
  double maxZ=2*std::max(std::fabs(maxExtent.z()),std::fabs(minExtent.z()));
  double extent=std::sqrt(maxX*maxX+maxY*maxY+maxZ*maxZ);
  double cell = 2*extent/ngrid;

  std::vector<UVector3> grid_points ;// new double[3*ngrid*ngrid*npercell];
  grid_points.resize(ngrid*ngrid*npercell);
  UVector3 point;
  UVector3 dir;
  double xmin, ymin;
  dir.x() = std::sin(theta*UUtils::kDegToRad)*std::cos(phi*UUtils::kDegToRad);
  dir.y() = std::sin(theta*UUtils::kDegToRad)*std::sin(phi*UUtils::kDegToRad);
  dir.z() = std::cos(theta*UUtils::kDegToRad);
  
#ifdef VECGEOM_ROOT
  int npoints = ngrid*ngrid*npercell;
  TPolyMarker3D *pmx = 0;
  TH2F *xprof = 0;
   if (graphics) {
      pmx = new TPolyMarker3D(npoints);
      pmx->SetMarkerColor(kRed);
      pmx->SetMarkerStyle(4);
      pmx->SetMarkerSize(0.2);
      xprof = new TH2F("x-ray",Form("X-ray profile from theta=%g phi=%g of shape %s", theta, phi, volumeUSolids->GetName().c_str()), 
                       ngrid, -extent, extent, ngrid, -extent, extent);
   }  
#endif
 
   //TGeoRotation *rot = new TGeoRotation("rot",phi-90,-theta,0);
   //TGeoCombiTrans *matrix = new TGeoCombiTrans(extent*dir[0], extent*dir[1], extent*dir[2], rot);
  
   UTransform3D* matrix= new UTransform3D(0,0,0,phi, theta, 0.);
   UVector3 origin = UVector3(extent*dir.x(),extent*dir.y(),extent*dir.z());
   
   dir=-dir;
 
   if ((fVerbose) && (graphics)) printf("=> x-ray direction:( %f, %f, %f)\n", dir.x(),dir.y(),dir.z());
   // loop cells   
   int ip=0;
   for (int i=0; i<ngrid; i++) {
      for (int j=0; j<ngrid; j++) {
         xmin = -extent + i*cell;
         ymin = -extent + j*cell;
         if (npercell==1) {
	   point.x() = xmin+0.5*cell;
	   point.y() = ymin+0.5*cell;
	   point.z() = 0;
	   grid_points[ip]=matrix->GlobalPoint(point)+origin;
	   //std::cout<<"ip="<<ip<<" grid="<<grid_points[ip]<<" xy="<<point.x()<<" "<<point.y()<<std::endl;
            #ifdef VECGEOM_ROOT
	   if (graphics) pmx->SetNextPoint(grid_points[ip].x(),grid_points[ip].y(),grid_points[ip].z());
            #endif
            ip++;
         } else {               
            for (int k=0; k<npercell; k++) {
	      point.x() = xmin+cell*vecgeom::RNG::Instance().uniform();
	      point.y() = ymin+cell*vecgeom::RNG::Instance().uniform();
	      point.z() = 0;
              grid_points[ip]= matrix->GlobalPoint(point)+origin;
	      //std::cout<<"ip="<<ip<<" grid="<<grid_points[ip]<<std::endl;
               #ifdef VECGEOM_ROOT
	       if (graphics) pmx->SetNextPoint(grid_points[ip].x(),grid_points[ip].y(),grid_points[ip].z());
               #endif
               ip++;
            }
         }
      }
   }
   double sum = 0;
   double sumerr = 0;
   double dist, lastdist;
   int nhit = 0;
   int ntransitions = 0;
   bool last = false;
   for (int i=0; i<ip; i++) {
      dist = CrossedLength(grid_points[i], dir, useeps);
      sum += dist;
      
      if (dist>0) {
         lastdist = dist;
         nhit++;
         if (!last) {
            ntransitions++;
            sumerr += lastdist;
         }   
         last = true;
         point=matrix->LocalPoint(grid_points[i]);
         #ifdef VECGEOM_ROOT
         if (graphics) {
	     xprof->Fill(point.x(), point.y(), dist);
         } 
         #endif  
      } else {
         if (last) {
            ntransitions++;
            sumerr += lastdist;
         }
         last = false;
      }
   }   
    gCapacitySampled = sum*cell*cell/npercell;
    gCapacityError = sumerr*cell*cell/npercell;
    gCapacityAnalytical =  volumeUSolids->Capacity();
    if((fVerbose) && (graphics)){
     printf("th=%g phi=%g: analytical: %f    --------   sampled: %f +/- %f\n", theta, phi, gCapacityAnalytical ,gCapacitySampled , gCapacityError);
     printf("Hit ratio: %f\n", double(nhit)/ip);
     if (nhit>0) printf("Average crossed length: %f\n", sum/nhit);
    }
   if((gCapacitySampled-gCapacityAnalytical)>10*gCapacityError) nError++;
  
#ifdef VECGEOM_ROOT
   if (graphics) {
     
     // new TCanvas("X-ray-test", "Shape and projection plane");
     
     // TGeoBBox *box = new TGeoBBox("box",5,5,5);
     //box->Draw();
     //pmx->Draw("SAME");

     //((TView3D*)gPad->GetView())->ShowAxis();
    
      new TCanvas("c11", "X-ray scan");
      xprof->DrawCopy("LEGO1");
      
   }
#endif

   if( nError )
     errCode= 512; //errCode: 0010 0000 0000

   return errCode;      
}
//////////////////////////////////////////////////////////////////////////////
double ShapeTester:: CrossedLength(const UVector3 &point, const UVector3 &dir, bool useeps)
{
// Return crossed length of the shape for the given ray, taking into account possible multiple crossings
   double eps = 0;
   
   if (useeps) eps = 1.E-9;
   double len = 0;
   double dist = volumeUSolids->DistanceToIn(point,dir);
   if (dist>1E10) return len;
   // Propagate from starting point with the found distance (on the numerical boundary)
   UVector3 pt(point),norm;
   bool convex;
  
   while (dist<1E10) {
      pt=pt+(dist+eps)*dir;    // ray entering
      // Compute distance from inside
      dist = volumeUSolids->DistanceToOut(pt,dir,norm,convex);
      len += dist;
      pt=pt+(dist+eps)*dir;     // ray exiting
      dist = volumeUSolids->DistanceToIn(pt,dir);
   }   
   return len;
}   
////////////////////////////////////////////////////////////////////////////
void ShapeTester::FlushSS(stringstream &ss)
{
	string s = ss.str();
	cout << s;
	*log << s;
	ss.str("");
}

void ShapeTester::Flush(const string &s)
{
	cout << s;
	*log << s;
}


// NEW: results written normalized to nano seconds per operation
double ShapeTester::NormalizeToNanoseconds(double time)
{
	double res = ((time * (double) 1e+9) / ((double)repeat * (double)maxPoints));
	return res;
}

double ShapeTester::MeasureTest (int (ShapeTester::*funcPtr)(int), const string &amethod)
{
	Flush("Measuring performance of method "+amethod+"\n");

	// NEW: storing data phase, timer is off
	// See: http://www.newty.de/fpt/fpt.html , The Function Pointer Tutorials
	(*this.*funcPtr)(0);

         clock_t start_t, end_t;

        start_t = clock();
	// performance phase, timer is on
	for (int i = 1; i <= repeat; i++)
	{
		(*this.*funcPtr)(i); 
	}

	end_t=clock();
        double realTime = end_t-start_t;

	stringstream ss;
	// NEW: write time per operation / bunch of operations
	ss << "Time elapsed: " << realTime << "s\n";
	ss << "Time per one repeat: " << realTime / repeat << "s\n";
	ss << "Nanoseconds per one method call: " << NormalizeToNanoseconds(realTime) << "\n";
	FlushSS(ss);

	return realTime;
}

void ShapeTester::CreatePointsAndDirectionsSurface()
{
  	UVector3 norm, point;   
	for (int i = 0; i < maxPointsSurface; i++)
	{
	
         UVector3 pointU;
         int retry = 100;
         do 
	   { bool surfaceExist=true;
	   if(surfaceExist) {pointU = volumeUSolids->GetPointOnSurface(); }
           else {
                UVector3 dir = GetRandomDirection(), norm;
                bool convex;
                double random=UUtils::Random();
                int index = (int)maxPointsInside*random;
                double dist = volumeUSolids->DistanceToOut(points[index],dir,norm,convex);
                pointU = points[index]+dir*dist ;

           }
           if (retry-- == 0) break;
         }
         while (volumeUSolids->Inside(pointU) != vecgeom::EInside::kSurface);

	UVector3 vec = GetRandomDirection();
	directions[i] = vec;
  	point.Set(pointU.x(), pointU.y(), pointU.z());
        points[i+offsetSurface] = point;
        
	}

}
void ShapeTester::CreatePointsAndDirectionsEdge()
{
	UVector3 norm, point; 
       
	for (int i = 0; i < maxPointsEdge; i++)
	{
	 UVector3 pointU;
         int retry = 100;
         do 
         {
	  volumeUSolids->SamplePointsOnEdge(1,&pointU);
          if (retry-- == 0) break;
         }
         while (volumeUSolids->Inside(pointU) != vecgeom::EInside::kSurface);
 	 UVector3 vec = GetRandomDirection();
	 directions[i] = vec;
   
	point.Set(pointU.x(), pointU.y(), pointU.z());
        points[i+offsetEdge] = point;
        
	}
     
}

void ShapeTester::CreatePointsAndDirectionsOutside()
{

	UVector3 minExtent,maxExtent;
    volumeUSolids->Extent(minExtent,maxExtent);
	double maxX=std::max(std::fabs(maxExtent.x()),std::fabs(minExtent.x()));
	double maxY=std::max(std::fabs(maxExtent.y()),std::fabs(minExtent.y()));
	double maxZ=std::max(std::fabs(maxExtent.z()),std::fabs(minExtent.z()));
    double rOut=std::sqrt(maxX*maxX+maxY*maxY+maxZ*maxZ);
        
    for (int i = 0; i < maxPointsOutside; i++)
	{
	          
	   UVector3 vec, point;
       do
	   {
	    point.x() =  -1 + 2 * UUtils::Random();
	    point.y() = -1 + 2 * UUtils::Random(); 
	    point.z() = -1 + 2 * UUtils::Random();
        point *= rOut * outsideMaxRadiusMultiple;
	   }
	   while (volumeUSolids->Inside(point)!=vecgeom::EInside::kOutside);


  	   double random = UUtils::Random();
	   if (random <= outsideRandomDirectionPercent/100.) 
	   {
		vec = GetRandomDirection();
	   }
	   else
	   {
		UVector3 pointSurface= volumeUSolids->GetPointOnSurface(); 
		vec = pointSurface - point;
		vec.Normalize();
	   }
		
	   points[i+offsetOutside] = point;
	   directions[i+offsetOutside] = vec;
	}

}

// DONE: inside points generation uses random points inside bounding box
void ShapeTester::CreatePointsAndDirectionsInside()
{       
  UVector3 minExtent,maxExtent;
  volumeUSolids->Extent(minExtent,maxExtent);
  int i = 0; 
  while (i < maxPointsInside)
  {
   double x = RandomRange(minExtent.x(), maxExtent.x());
   double y = RandomRange(minExtent.y(), maxExtent.y());
   if (minExtent.y() == maxExtent.y())
   y = RandomRange(-1000, +1000);
   double z = RandomRange(minExtent.z(), maxExtent.z());
   UVector3 point0(x, y, z);
   if (volumeUSolids->Inside(point0)==vecgeom::EInside::kInside)
   {               
    UVector3 point(x, y, z);
    UVector3 vec = GetRandomDirection();
    points[i+offsetInside] = point;
    directions[i+offsetInside] = vec;
    i++;
   }
  }
}

void ShapeTester::CreatePointsAndDirections()
{ 
  if(method != "XRayProfile")
    {
    maxPointsInside = (int) (maxPoints * (insidePercent/100));
    maxPointsOutside = (int) (maxPoints * (outsidePercent/100));
    maxPointsEdge = (int) (maxPoints * (edgePercent/100));
    maxPointsSurface = maxPoints - maxPointsInside - maxPointsOutside-maxPointsEdge;
      
    offsetInside = 0;
    offsetSurface = maxPointsInside;
    offsetEdge = offsetSurface + maxPointsSurface;
    offsetOutside = offsetEdge+maxPointsEdge;

    points.resize(maxPoints);
    directions.resize(maxPoints);
    resultDoubleDifference.resize(maxPoints);
    resultBoolUSolids.resize(maxPoints);
    resultDoubleUSolids.resize(maxPoints);

    resultVectorDifference.resize(maxPoints);
    resultVectorUSolids.resize(maxPoints);

    CreatePointsAndDirectionsOutside();
    CreatePointsAndDirectionsInside();
    CreatePointsAndDirectionsSurface();
    }
}


#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().


int directoryExists (string s)
{
  {
   struct stat status;
   stat(s.c_str(), &status);
   return (status.st_mode & S_IFDIR);
  }
  return false;
}


void ShapeTester::PrintCoordinates (stringstream &ss, const UVector3 &vec, const string &delimiter, int precision)
{ 
	ss.precision(precision);
	ss << vec.x() << delimiter << vec.y() << delimiter << vec.z();
}

string ShapeTester::PrintCoordinates (const UVector3 &vec, const string &delimiter, int precision)
{
	static stringstream ss;
	PrintCoordinates(ss, vec, delimiter, precision);
	string res(ss.str());
	ss.str("");
	return res;
}

string ShapeTester::PrintCoordinates (const UVector3 &vec, const char *delimiter, int precision)
{
	string d(delimiter);
	return PrintCoordinates(vec, d, precision);
}

void ShapeTester::PrintCoordinates (stringstream &ss, const UVector3 &vec, const char *delimiter, int precision)
{
	string d(delimiter);
	return PrintCoordinates(ss, vec, d, precision);
}


int ShapeTester::CountDoubleDifferences(const vector<double> &differences, const vector<double> &values1, const vector<double> &values2)	 
{
	int countOfDifferences = 0;
	stringstream ss;

	for (int i = 0; i < maxPoints; i++) 
	{
		double value1 = values1[i];
		double value2 = values2[i];
		double dif = differences[i];
		double difference = std::abs (dif);
		if (difference > std::abs (differenceTolerance*value1))
		{
			if (++countOfDifferences <= 10) ss << "Different point found: index " << i << 
				"; point coordinates:" << PrintCoordinates(points[i], ",") << 
				"; direction coordinates:" << PrintCoordinates(directions[i], ",") <<
				"; difference=" << difference << ")" << 
				"; value2 =" << value2 <<
				"; value1 = " << value1 << "\n";
		}
	}
	ss << "Number of differences is " << countOfDifferences << "\n";
	FlushSS(ss);
	return countOfDifferences;
}

int ShapeTester::CountDoubleDifferences(const vector<double> &differences)
{
	int countOfDifferences = 0;

	for (int i = 0; i < maxPoints; i++) 
	{
		double difference = std::abs (differences[i]);
		if (difference > differenceTolerance) countOfDifferences++;
	}
	stringstream ss;
	ss << "Number of differences is " << countOfDifferences << "\n";
	FlushSS(ss); 
	return countOfDifferences;
}

// NEW: output values precision setprecision (16)
// NEW: for each method, one file

// NEW: print also different point coordinates

void ShapeTester::VectorToDouble(const vector<UVector3> &vectorUVector, vector<double> &vectorDouble)
{
	UVector3 vec;

	int size = vectorUVector.size();
	for (int i = 0; i < size; i++)
	{
		vec = vectorUVector[i];
		double mag = vec.Mag();
		if (mag > 1.1) 
			mag = 1;
		vectorDouble[i] = mag;
	}
}

void ShapeTester::BoolToDouble(const std::vector<bool> &vectorBool, std::vector<double> &vectorDouble)
{
  int size = vectorBool.size();
  for (int i = 0; i < size; i++)
    vectorDouble[i] = (double) vectorBool[i];
}

int ShapeTester::SaveResultsToFile(const string &method1)
{
        string name=volumeUSolids->GetName();
	string filename1(folder+name+"_"+method1+".dat");
	std::cout<<"Saving all results to " <<filename1 <<std::endl;
	ofstream file(filename1.c_str());
	bool saveVectors = (method1 == "Normal");
	int prec = 16;
	if (file.is_open())
	{
		file.precision(prec);
		file << volumeString << "\n";
		string spacer("\t");
		for (int i = 0; i < maxPoints; i++)
		{
			
			file << PrintCoordinates(points[i], spacer, prec) << spacer << PrintCoordinates(directions[i], spacer, prec) << spacer; 
			if (saveVectors) file << PrintCoordinates(resultVectorUSolids[i], spacer, prec) << "\n";
			else file <<  resultDoubleUSolids[i] << "\n";
		}
		return 0;
	}
	std::cout<<"Unable to create file "<<filename1<<std::endl;
	return 1;
}

int ShapeTester::TestMethod(int (ShapeTester::*funcPtr)())
{
  int errCode= 0;

  std::cout<< "========================================================= "<<std::endl;

  if(method != "XRayProfile"){
    std::cout<< "% Creating " <<  maxPoints << " points and directions for method =" <<method<<std::endl;
    
    CreatePointsAndDirections();
    cout.precision(20);
    std::cout<< "% Statistics: points=" << maxPoints << ",\n";

    std::cout << "%             ";
    std::cout << "surface=" << maxPointsSurface << ", inside=" << maxPointsInside << ", outside=" <<   
      maxPointsOutside << "\n";
  }
  std::cout << "%     "<<std::endl;
   

  errCode+= (*this.*funcPtr)();
  std::cout<< "========================================================= "<<std::endl;

  return errCode;
}

//will run all tests. in this case, one file stream will be used
int ShapeTester::TestMethodAll()
{   
  int errCode=0;

  method = "Consistency";
  errCode+= TestMethod(&ShapeTester::TestConsistencySolids);
  if(definedNormal) TestMethod(&ShapeTester::TestNormalSolids);
  method = "SafetyFromInside";
  errCode+= TestMethod(&ShapeTester::TestSafetyFromInsideSolids);
  method = "SafetyFromOutside";
  errCode+= TestMethod(&ShapeTester::TestSafetyFromOutsideSolids);
  method = "DistanceToIn";
  errCode+= TestMethod(&ShapeTester::TestDistanceToInSolids);
  method = "DistanceToOut";
  errCode+= TestMethod(&ShapeTester::TestDistanceToOutSolids);
  method = "XRayProfile";
  errCode+= TestMethod(&ShapeTester::TestXRayProfile);


  method = "all";

  return errCode;
}

void ShapeTester::SetFolder(const string &newFolder)
{
   cout << "Checking for existance of " << newFolder << endl;
      
   if (!directoryExists(newFolder))
   {
	string command;
	#ifdef WIN32
		_mkdir(newFolder.c_str());
	#else
		std::cout<<"try to create dir for "<<std::endl;
		mkdir(newFolder.c_str(), 0777);
	#endif
	if (!directoryExists(newFolder))
	{
		cout << "Directory "+newFolder+" does not exist, it must be created first\n";
		exit(1);
	}

        }
	folder = newFolder+"/";

}

int ShapeTester::Run(VUSolid *testVolume)
{
  int errCode= 0;
  stringstream ss;

  int (ShapeTester::*funcPtr)()=NULL;
  
  volumeUSolids= testVolume;
  std::ofstream logger("/log/box");
  log = &logger;
  
  SetFolder("log");
  
  if (method == "") method = "all";
  string name = testVolume->GetName();
  std::cout<< "\n\n";
  std::cout << "===============================================================================\n";
  std::cout << "Invoking test for method " << method << " on " << name << " ..." << "\nFolder is " << folder << std::endl;
  std::cout<< "===============================================================================\n";
  std::cout<< "\n";
  
  if (method == "Consistency") funcPtr = &ShapeTester::TestConsistencySolids;
  if (method == "Normal") funcPtr = &ShapeTester::TestNormalSolids;
  if (method == "SafetyFromInside") funcPtr = &ShapeTester::TestSafetyFromInsideSolids;
  if (method == "SafetyFromOutside") funcPtr = &ShapeTester::TestSafetyFromOutsideSolids;
  if (method == "DistanceToIn") funcPtr = &ShapeTester::TestDistanceToInSolids;
  if (method == "DistanceToOut") funcPtr = &ShapeTester::TestDistanceToOutSolids;
  if (method == "XRayProfile") funcPtr = &ShapeTester::TestXRayProfile;
  
  if (method == "all") errCode+= TestMethodAll();
  else if (funcPtr) errCode+= TestMethod(funcPtr);
  else std::cout<< "Method " << method << " is not supported" << std::endl;
  
  ClearErrors();
  method = "all";

  return errCode;
}
int ShapeTester::RunMethod(VUSolid *testVolume, std::string method1)
{
  int errCode= 0;
  stringstream ss;

  int (ShapeTester::*funcPtr)()=NULL;
  
  volumeUSolids= testVolume;
  std::ofstream logger("/log/box");
  log = &logger;
  
  SetFolder("log");
  
  method = method1;
  
  if (method == "") method = "all";
  string name = testVolume->GetName();
  
  std::cout<< "\n\n";
  std::cout << "===============================================================================\n";
  std::cout << "Invoking test for method " << method << " on " << name << " ..." << "\nFolder is " << folder << std::endl;
  std::cout<< "===============================================================================\n";
  std::cout<< "\n";
  
  if (method == "Consistency") funcPtr = &ShapeTester::TestConsistencySolids;
  if (method == "Normal") funcPtr = &ShapeTester::TestNormalSolids;
  if (method == "SafetyFromInside") funcPtr = &ShapeTester::TestSafetyFromInsideSolids;
  if (method == "SafetyFromOutside") funcPtr = &ShapeTester::TestSafetyFromOutsideSolids;
  if (method == "DistanceToIn") funcPtr = &ShapeTester::TestDistanceToInSolids;
  if (method == "DistanceToOut") funcPtr = &ShapeTester::TestDistanceToOutSolids;
  
  if (method == "XRayProfile") funcPtr = &ShapeTester::TestXRayProfile;
  if (method == "all") errCode+= TestMethodAll();
  else if (funcPtr) errCode+= TestMethod(funcPtr);
  else std::cout<< "Method " << method << " is not supported" << std::endl;
  
  ClearErrors();
  method = "all";

  return errCode;
}
//
// ReportError
//
// Report the specified error message, but only if it has not been reported a zillion
// times already.
//
void ShapeTester::ReportError( int *nError,  UVector3 &p, 
			   UVector3 &v, double distance,
			       std::string comment)//, std::ostream &logger )
{
  
  ShapeTesterErrorList *last=0, *errors = errorList;
  while( errors ) {
    
    if (errors->message == comment) {
      if ( ++errors->nUsed > 5 ) return;
      break;
    }
    last = errors;
    errors = errors->next;
  }

  if (errors == 0) {
    //
    // New error: add it the end of our list
    //
    errors = new ShapeTesterErrorList;
    errors->message = comment;
    errors->nUsed = 1;
    errors->next = 0;
    if (errorList) 
      last->next = errors;
    else
      errorList = errors;
  }

  //
  // Output the message
  //	
 
  std::cout << "% " << comment;
  if (errors->nUsed == 5) std::cout << " (any further such errors suppressed)";
  std::cout << " Distance = " << distance ; 
  std::cout << std::endl;

  std::cout << ++(*nError) << " " << p.x() << " " << p.y() << " " << p.z() 
	    << " " << v.x() << " " << v.y() << " " << v.z() << std::endl;
  
  //
  // if debugging mode we have to exit now
  //
  if(ifException){
     std::ostringstream text;
     text << "Abborting due to Debugging mode in solid: " << volumeUSolids->GetName();
     UUtils::Exception("ShapeTester", "Debugging mode", UFatalErrorInArguments, 1, text.str().c_str());
  }
}
//
// ClearErrors
// Reset list of errors (and clear memory)
//
void ShapeTester::ClearErrors()
{
  ShapeTesterErrorList *here, *next;

  here = errorList;
  while( here ) {
    next = here->next;
    delete here;
    here = next;
  }
  errorList = 0;
}
//
// CountErrors
//
int ShapeTester::CountErrors() const
{
  ShapeTesterErrorList *here;
  int answer = 0;

  here = errorList;
  while( here ) {
    answer += here->nUsed;
    here = here->next;
  }

  return answer;
}
