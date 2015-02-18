#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UTrd.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Trd.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//typedef UCons Cone_t;
typedef vecgeom::SimpleTrd Trd_t;

int main(int argc,char *argv[]) {
  int errCode= 0;
  VUSolid* solid;
  if(argc>1){
    if(strcmp(argv[1],"vec")==0)
    {
      solid=new Trd_t("test_VecGeomTrd",5.,5. ,6.,6.,5.);    
    }
    else
    {   
     solid=new UTrd("test_USolidsTrd",5.,5.,6.,6.,5.);  
     solid->StreamInfo(std::cout);      
    }
  }
  else
    {
     solid=new UTrd("test_USolidsTrd",5.,5.,6.,6.,5.);  
     solid->StreamInfo(std::cout);
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



