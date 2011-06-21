#ifndef ROOT_TGeoUShape
#define ROOT_TGeoUShape

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoUShape - bridge class for using a USolid as TGeoShape.             //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class VUSolid;

class TGeoUShape : public TGeoBBox
{
private:
   VUSolid            *fUSolid;      // Unified solid

public:
   TGeoUShape() : TGeoBBox(), fUSolid(NULL) {}
   TGeoUShape(const char *name, VUSolid *solid) : TGeoBBox(name,0,0,0), fUSolid(solid) {}   
   virtual ~TGeoUShape();
   
// Navigation
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
// Visualization and overlap checking
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const;
//   virtual void          SetPoints(Double_t *points) const;
//   virtual void          SetSegsAndPols(TBuffer3D &buffer) const;

// Auxiliary
   virtual Double_t      Capacity() const;

// Static creators for USolids
   static VUSolid       *CreateBox(Double_t dx, Double_t dy, Double_t dz);
   static VUSolid       *CreateTube(Double_t dz, Double_t rmin, Double_t rmax, Double_t phi0, Double_t dphi);
//   ...

   ClassDef(TGeoUShape, 1)         // an external USolid
};
#endif
