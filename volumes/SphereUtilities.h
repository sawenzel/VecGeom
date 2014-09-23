#ifndef VECGEOM_VOLUMES_SPHEREUTILITIES_H_
#define VECGEOM_VOLUMES_SPHEREUTILITIES_H_

#include "base/Global.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include "base/RNG.h"

namespace VECGEOM_NAMESPACE {

  
  
  
      
  VECGEOM_CUDA_HEADER_BOTH
  Precision sqr(Precision x) {return x*x;}; 
  
  /*
  template <class Backend>  
  VECGEOM_CUDA_HEADER_BOTH
  void fabs(typename Backend::precision_v &v)
  {
      typedef typename Backend::precision_v Double_t;
      Double_t mone(-1.);
      MaskedAssign( (v<0), mone*v , &v );
  }
  */
  
  VECGEOM_CUDA_HEADER_BOTH
  Precision GetRadiusInRing(Precision rmin, Precision rmax)
  {
  // Generate radius in annular ring according to uniform area
  //
  if (rmin <= 0.)
  {
    return rmax * std::sqrt(RNG::Instance().uniform(0. , 1.));
  }
  if (rmin != rmax)
  {
    return std::sqrt(RNG::Instance().uniform(0. , 1.)
                     * (sqr(rmax) - sqr(rmin)) + sqr(rmin));
  }
  return rmin;
}
  
  
  

}// End global 

#endif //VECGEOM_VOLUMES_SPHEREUTILITIES_H_