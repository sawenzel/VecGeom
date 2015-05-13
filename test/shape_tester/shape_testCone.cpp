#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UCons.hh"

#include "base/Vector3D.h"
#include "base/Global.h"
#include "volumes/Cone.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//typedef UCons Cone_t;
typedef vecgeom::SimpleCone Cone_t;

int main(int argc,char *argv[]) {
 int errCode= 0;
 VUSolid* cone;
  if(argc>1){
    if(strcmp(argv[1],"vec")==0)
    {
     cone=new Cone_t("test_VecGeomCone",5.,6.,5.5,7.,2,0,vecgeom::kTwoPi);
    
    }
    else
    {   
     cone=new UCons("test_USolidsCone",5.,6.,5.5,7.,7.,0,vecgeom::kTwoPi*0.3);
     cone->StreamInfo(std::cout);  
    }
  }
  else
    {
     cone=new UCons("test_USolidsCone",5.,6.,5.5,7.,7.,0,vecgeom::kTwoPi*0.3);
     cone->StreamInfo(std::cout);
    }
  
  ShapeTester tester;

  if(argc>2)
  {
    if(strcmp(argv[2],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     errCode = tester.Run(cone);
     theApp.Run();
     #endif
    }
  }
  else
  {
    errCode = tester.Run(cone);
   }
   std::cout<<"Final Error count for Shape *** "<<cone->GetName()<<"*** = "<<errCode<<std::endl;
  std::cout<<"========================================================="<<std::endl;
  return 0;
}

