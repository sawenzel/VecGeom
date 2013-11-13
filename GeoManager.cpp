/*
 * GeoManager.cpp
 *
 *  Created on: Nov 12, 2013
 *      Author: swenzel
 */

#include "GeoManager.h"
#include "PhysicalBox.h"
#include "TransformationMatrix.h"

/*
template<int tid, int rid>
PhysicalVolume * dosomething(int i, int j, BoxParameters *bp, TransformationMatrix *tm)
{
	if(i==tid && j==rid) return new PlacedBox<tid, rid>(bp, tm);
	return 0;
}

template<int upperx, int uppery, const int MAXY>
class EmitCode
{
public:
  static
  PhysicalVolume* emit(int i, int j, BoxParameters *bp, TransformationMatrix *tm)
  {
    dosomething<upperx,uppery>(i,j,bp,tm);
    EmitCode<upperx, uppery-1, MAXY>::emit(i,j, bp, tm);
  }
};


// template specialization
template<const int MAXY>
class EmitCode<0,0,MAXY>
{
public:
  static
  PhysicalVolume* emit(int i, int j, BoxParameters *bp, TransformationMatrix *tm)
  {
    dosomething<0,0>(i,j, bp, tm);
  }
};


// template specialization
template<int upperx, const int MAXY>
class EmitCode<upperx,0,MAXY>
{
public:
  static
  PhysicalVolume* emit(int i, int j, BoxParameters *bp, TransformationMatrix *tm)
  {
    dosomething<upperx,0>(i,j, bp, tm);
    EmitCode<upperx-1, MAXY, MAXY>::emit(i,j, bp, tm);
  }
};


// template specialization
template<int uppery, const int MAXY>
class EmitCode<0,uppery,MAXY>
{
public:
  static
  PhysicalVolume* emit(int i, int j, BoxParameters *bp, TransformationMatrix *tm)
  {
    dosomething<0,uppery>(i, j, bp, tm);
    EmitCode<0, uppery-1, MAXY>::emit(i,j,bp,tm);
  }
};
*/

PhysicalVolume * MakePlacedBox(BoxParameters const * bp, TransformationMatrix const * tm)
{
	// get footprint of TransformationMatrix
	int rid = tm->getRotationFootprint();
	int tid = tm->GetTranslationIdType();

	// the following piece of code is script generated

	if(tid == 0 && rid == 0 ) return new PlacedBox<0,0>(bp,tm); // identity
	if(tid == 1 && rid == 0 ) return new PlacedBox<1,0>(bp,tm); // identity
	if(tid == 0 && rid == 252 ) return new PlacedBox<0,252>(bp,tm);
	if(tid == 1 && rid == 252 ) return new PlacedBox<1,252>(bp,tm);
	if(tid == 0 && rid == 405 ) return new PlacedBox<0,405>(bp,tm);
	if(tid == 1 && rid == 405 ) return new PlacedBox<1,405>(bp,tm);
	if(tid == 0 && rid == 882 ) return new PlacedBox<0,882>(bp,tm);
	if(tid == 1 && rid == 882 ) return new PlacedBox<1,882>(bp,tm);
	if(tid == 0 && rid == 415 ) return new PlacedBox<0,415>(bp,tm);
	if(tid == 1 && rid == 415 ) return new PlacedBox<1,415>(bp,tm);
	if(tid == 0 && rid == 496 ) return new PlacedBox<0,496>(bp,tm);
	if(tid == 1 && rid == 496 ) return new PlacedBox<1,496>(bp,tm);
	if(tid == 0 && rid == 793 ) return new PlacedBox<0,793>(bp,tm);
	if(tid == 1 && rid == 793 ) return new PlacedBox<1,793>(bp,tm);
	if(tid == 0 && rid == 638 ) return new PlacedBox<0,638>(bp,tm);
	if(tid == 1 && rid == 638 ) return new PlacedBox<1,638>(bp,tm);
	if(tid == 0 && rid == 611 ) return new PlacedBox<0,611>(bp,tm);
	if(tid == 1 && rid == 611 ) return new PlacedBox<1,611>(bp,tm);
	if(tid == 0 && rid == 692 ) return new PlacedBox<0,692>(bp,tm);
	if(tid == 1 && rid == 692 ) return new PlacedBox<1,692>(bp,tm);
	if(tid == 0 && rid == 720 ) return new PlacedBox<0,720>(bp,tm);
	if(tid == 1 && rid == 720 ) return new PlacedBox<1,720>(bp,tm);
	if(tid == 0 && rid == 828 ) return new PlacedBox<0,828>(bp,tm);
	if(tid == 1 && rid == 828 ) return new PlacedBox<1,828>(bp,tm);
	if(tid == 0 && rid == 756 ) return new PlacedBox<0,756>(bp,tm);
	if(tid == 1 && rid == 756 ) return new PlacedBox<1,756>(bp,tm);
	if(tid == 0 && rid == 918 ) return new PlacedBox<0,918>(bp,tm);
	if(tid == 1 && rid == 918 ) return new PlacedBox<1,918>(bp,tm);
	if(tid == 0 && rid == 954 ) return new PlacedBox<0,954>(bp,tm);
	if(tid == 1 && rid == 954 ) return new PlacedBox<1,954>(bp,tm);
	if(tid == 0 && rid == 1008 ) return new PlacedBox<0,1008>(bp,tm);
	if(tid == 1 && rid == 1008 ) return new PlacedBox<1,1008>(bp,tm);
	// fallback case
	return new PlacedBox<-1,-1>(bp,tm);
}
