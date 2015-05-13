#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTrap.hh"

#include "base/Vector3D.h"
#include "volumes/Trapezoid.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//using Trap_t = UTrap;
using Trap_t = vecgeom::SimpleTrapezoid;

int main(int argc,char *argv[]) {
  int errCode= 0;
  VUSolid* solid;
  if(argc>1){
    if(strcmp(argv[1],"vec")==0)
    {
      solid = new Trap_t("test_VecGeomTrap",5.,0.,0.,5.,5.,5.,0.,5.,5.,5.,0.);   
    }
    else
    {   
     solid = new UTrap("test_USolidsTrap",40,0,0,30,20,20,0,30,20,20,0);  
     solid->StreamInfo(std::cout);      
    }
  }
  else
    {
     solid = new Trap_t("test_VecGeomTrap",5.,0.,0.,5.,5.,5.,0.,5.,5.,5.,0.);    
    }
  
  ShapeTester tester;

  if(argc>2)
  {
    if(strcmp(argv[2],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     errCode = tester.Run(solid);
     theApp.Run();
     #endif
    }
  }
  else
  {
    errCode = tester.Run(solid);
   }
  std::cout<<"Final Error count for Shape *** "<<solid->GetName()<<"*** = "<<errCode<<std::endl;
  std::cout<<"========================================================="<<std::endl;
  return 0;
}
