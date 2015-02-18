#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTubs.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Tube.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

typedef vecgeom::SimpleTube Tube_t;

int main(int argc,char *argv[]) {
  int errCode= 0;
  VUSolid* tube;
  if(argc>1){
    if(strcmp(argv[1],"vec")==0)
    {
     tube=new Tube_t("test_VecGeomTube",0.,6.,2,0,vecgeom::kTwoPi); 
    
    }
    else
    { 
     tube=new UTubs("test_USolidsTube",1.,6.,2,0,vecgeom::kTwoPi*0.6);
     tube->StreamInfo(std::cout);
    }
  }
  else
    {
     tube=new UTubs("test_USolidsTube",1.,6.,2,0,vecgeom::kTwoPi*0.6);
     tube->StreamInfo(std::cout);
    }
  
  ShapeTester tester;

  if(argc>2)
  {
    if(strcmp(argv[2],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     errCode = tester.Run(tube);
     theApp.Run();
     #endif
    }
  }
  else
  {
    errCode = tester.Run(tube);
  }
   std::cout<<"Final Error count for Shape *** "<<tube->GetName()<<"*** = "<<errCode<<std::endl;
  std::cout<<"========================================================="<<std::endl;
  return 0;
}



