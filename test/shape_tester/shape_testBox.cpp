#include "ShapeTester.h"
#include "VUSolid.hh"
#include "UBox.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "TApplication.h"
#endif
#include "stdlib.h"

//typedef UBox Box_t;
typedef vecgeom::SimpleBox Box_t;

int main(  int argc,char *argv[]) {

 int errCode= 0;
 VUSolid* box;
  if(argc>1){
    if(strcmp(argv[1],"vec")==0)
    {
     box=new Box_t("test_VecGeomBox",5.,5.,5.);
    }
    else
    {   
     box=new UBox("test_USolidsBox",5.,5.,5.);
     box->StreamInfo(std::cout);
    }
  }
  else
    {
     box=new Box_t("test_VecGeomBox",5.,5.,5.);
    
    }
  ShapeTester tester;

  if(argc>2)
  {
    if(strcmp(argv[2],"vis")==0)
    {
     #ifdef VECGEOM_ROOT
     TApplication theApp("App",0,0);
     errCode = tester.Run(box);
     theApp.Run();
     #endif
    }
  }
  else
  {
    errCode = tester.Run(box);

   }
  std::cout<<"Final Error count for Shape *** "<<box->GetName()<<"*** = "<<errCode<<std::endl;
  std::cout<<"========================================================="<<std::endl;
  return 0;
}






