/*
 * GeoManager.cpp
 *
 *  Created on: Nov 12, 2013
 *      Author: swenzel
 */

#include "GeoManager.h"
#include "PhysicalBox.h"
#include "PhysicalTube.h"
#include "TransformationMatrix.h"
#include "PhysicalVolume.h"


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

PhysicalVolume * GeoManager::MakePlacedBox(BoxParameters const * bp, TransformationMatrix const * tm)
{
	// get footprint of TransformationMatrix
	int rid = tm->getRotationFootprint();
	int tid = tm->GetTranslationIdType();

	// the following piece of code is script generated
	if(tid == 0 && rid == 1296 ) return new PlacedBox<0,1296>(bp,tm); // identity
	if(tid == 1 && rid == 1296 ) return new PlacedBox<1,1296>(bp,tm); // identity
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

	// these just force code bloat
#ifdef LIKETOHAVECODEBLOAT
	if( tid == 10 && rid == 20 ) return new PlacedBox<10,20>(bp,tm);
	if( tid == 10 && rid == 21 ) return new PlacedBox<10,21>(bp,tm);
	if( tid == 10 && rid == 22 ) return new PlacedBox<10,22>(bp,tm);
	if( tid == 10 && rid == 23 ) return new PlacedBox<10,23>(bp,tm);
	if( tid == 10 && rid == 24 ) return new PlacedBox<10,24>(bp,tm);
	if( tid == 10 && rid == 25 ) return new PlacedBox<10,25>(bp,tm);
	if( tid == 10 && rid == 26 ) return new PlacedBox<10,26>(bp,tm);
	if( tid == 10 && rid == 27 ) return new PlacedBox<10,27>(bp,tm);
	if( tid == 10 && rid == 28 ) return new PlacedBox<10,28>(bp,tm);
	if( tid == 10 && rid == 29 ) return new PlacedBox<10,29>(bp,tm);
	if( tid == 10 && rid == 30 ) return new PlacedBox<10,30>(bp,tm);
	if( tid == 10 && rid == 31 ) return new PlacedBox<10,31>(bp,tm);
	if( tid == 10 && rid == 32 ) return new PlacedBox<10,32>(bp,tm);
	if( tid == 10 && rid == 33 ) return new PlacedBox<10,33>(bp,tm);
	if( tid == 10 && rid == 34 ) return new PlacedBox<10,34>(bp,tm);
	if( tid == 10 && rid == 35 ) return new PlacedBox<10,35>(bp,tm);
	if( tid == 10 && rid == 36 ) return new PlacedBox<10,36>(bp,tm);
	if( tid == 10 && rid == 37 ) return new PlacedBox<10,37>(bp,tm);
	if( tid == 10 && rid == 38 ) return new PlacedBox<10,38>(bp,tm);
	if( tid == 10 && rid == 39 ) return new PlacedBox<10,39>(bp,tm);
	if( tid == 10 && rid == 40 ) return new PlacedBox<10,40>(bp,tm);
	if( tid == 20 && rid == 20 ) return new PlacedBox<20,20>(bp,tm);
	if( tid == 20 && rid == 21 ) return new PlacedBox<20,21>(bp,tm);
	if( tid == 20 && rid == 22 ) return new PlacedBox<20,22>(bp,tm);
	if( tid == 20 && rid == 23 ) return new PlacedBox<20,23>(bp,tm);
	if( tid == 20 && rid == 24 ) return new PlacedBox<20,24>(bp,tm);
	if( tid == 20 && rid == 25 ) return new PlacedBox<20,25>(bp,tm);
	if( tid == 20 && rid == 26 ) return new PlacedBox<20,26>(bp,tm);
	if( tid == 20 && rid == 27 ) return new PlacedBox<20,27>(bp,tm);
	if( tid == 20 && rid == 28 ) return new PlacedBox<20,28>(bp,tm);
	if( tid == 20 && rid == 29 ) return new PlacedBox<20,29>(bp,tm);
	if( tid == 20 && rid == 30 ) return new PlacedBox<20,30>(bp,tm);
	if( tid == 20 && rid == 31 ) return new PlacedBox<20,31>(bp,tm);
	if( tid == 20 && rid == 32 ) return new PlacedBox<20,32>(bp,tm);
	if( tid == 20 && rid == 33 ) return new PlacedBox<20,33>(bp,tm);
	if( tid == 20 && rid == 34 ) return new PlacedBox<20,34>(bp,tm);
	if( tid == 20 && rid == 35 ) return new PlacedBox<20,35>(bp,tm);
	if( tid == 20 && rid == 36 ) return new PlacedBox<20,36>(bp,tm);
	if( tid == 20 && rid == 37 ) return new PlacedBox<20,37>(bp,tm);
	if( tid == 20 && rid == 38 ) return new PlacedBox<20,38>(bp,tm);
	if( tid == 20 && rid == 39 ) return new PlacedBox<20,39>(bp,tm);
	if( tid == 20 && rid == 40 ) return new PlacedBox<20,40>(bp,tm);
#endif

	// fallback case
	return new PlacedBox<1,-1>(bp,tm);
}


// already here we have repeatable pattern
// it would be nicer to have template factory
// MakeVolume<T>( Parameters<T>::Type const *, TransformationMatrix const * )
PhysicalVolume * GeoManager::MakePlacedTube( TubeParameters<> const * bp, TransformationMatrix const * tm )
{
	// get footprint of TransformationMatrix
	int rid = tm->getRotationFootprint();
	int tid = tm->GetTranslationIdType();

	// the following piece of code is script generated
	if(tid == 0 && rid == 1296 ) return new PlacedUSolidsTube<0,1296>(bp,tm); // identity
	if(tid == 1 && rid == 1296 ) return new PlacedUSolidsTube<1,1296>(bp,tm); // identity
	if(tid == 0 && rid == 252 ) return new PlacedUSolidsTube<0,252>(bp,tm);
	if(tid == 1 && rid == 252 ) return new PlacedUSolidsTube<1,252>(bp,tm);
	if(tid == 0 && rid == 405 ) return new PlacedUSolidsTube<0,405>(bp,tm);
	if(tid == 1 && rid == 405 ) return new PlacedUSolidsTube<1,405>(bp,tm);
	if(tid == 0 && rid == 882 ) return new PlacedUSolidsTube<0,882>(bp,tm);
	if(tid == 1 && rid == 882 ) return new PlacedUSolidsTube<1,882>(bp,tm);
	if(tid == 0 && rid == 415 ) return new PlacedUSolidsTube<0,415>(bp,tm);
	if(tid == 1 && rid == 415 ) return new PlacedUSolidsTube<1,415>(bp,tm);
	if(tid == 0 && rid == 496 ) return new PlacedUSolidsTube<0,496>(bp,tm);
	if(tid == 1 && rid == 496 ) return new PlacedUSolidsTube<1,496>(bp,tm);
	if(tid == 0 && rid == 793 ) return new PlacedUSolidsTube<0,793>(bp,tm);
	if(tid == 1 && rid == 793 ) return new PlacedUSolidsTube<1,793>(bp,tm);
	if(tid == 0 && rid == 638 ) return new PlacedUSolidsTube<0,638>(bp,tm);
	if(tid == 1 && rid == 638 ) return new PlacedUSolidsTube<1,638>(bp,tm);
	if(tid == 0 && rid == 611 ) return new PlacedUSolidsTube<0,611>(bp,tm);
	if(tid == 1 && rid == 611 ) return new PlacedUSolidsTube<1,611>(bp,tm);
	if(tid == 0 && rid == 692 ) return new PlacedUSolidsTube<0,692>(bp,tm);
	if(tid == 1 && rid == 692 ) return new PlacedUSolidsTube<1,692>(bp,tm);
	if(tid == 0 && rid == 720 ) return new PlacedUSolidsTube<0,720>(bp,tm);
	if(tid == 1 && rid == 720 ) return new PlacedUSolidsTube<1,720>(bp,tm);
	if(tid == 0 && rid == 828 ) return new PlacedUSolidsTube<0,828>(bp,tm);
	if(tid == 1 && rid == 828 ) return new PlacedUSolidsTube<1,828>(bp,tm);
	if(tid == 0 && rid == 756 ) return new PlacedUSolidsTube<0,756>(bp,tm);
	if(tid == 1 && rid == 756 ) return new PlacedUSolidsTube<1,756>(bp,tm);
	if(tid == 0 && rid == 918 ) return new PlacedUSolidsTube<0,918>(bp,tm);
	if(tid == 1 && rid == 918 ) return new PlacedUSolidsTube<1,918>(bp,tm);
	if(tid == 0 && rid == 954 ) return new PlacedUSolidsTube<0,954>(bp,tm);
	if(tid == 1 && rid == 954 ) return new PlacedUSolidsTube<1,954>(bp,tm);
	if(tid == 0 && rid == 1008 ) return new PlacedUSolidsTube<0,1008>(bp,tm);
	if(tid == 1 && rid == 1008 ) return new PlacedUSolidsTube<1,1008>(bp,tm);

#ifdef LIKETOHAVECODEBLOAT
	if( tid == 10 && rid == 20 ) return new PlacedUSolidsTube<10,20>(bp,tm);
	if( tid == 10 && rid == 21 ) return new PlacedUSolidsTube<10,21>(bp,tm);
	if( tid == 10 && rid == 22 ) return new PlacedUSolidsTube<10,22>(bp,tm);
	if( tid == 10 && rid == 23 ) return new PlacedUSolidsTube<10,23>(bp,tm);
	if( tid == 10 && rid == 24 ) return new PlacedUSolidsTube<10,24>(bp,tm);
	if( tid == 10 && rid == 25 ) return new PlacedUSolidsTube<10,25>(bp,tm);
	if( tid == 10 && rid == 26 ) return new PlacedUSolidsTube<10,26>(bp,tm);
	if( tid == 10 && rid == 27 ) return new PlacedUSolidsTube<10,27>(bp,tm);
	if( tid == 10 && rid == 28 ) return new PlacedUSolidsTube<10,28>(bp,tm);
	if( tid == 10 && rid == 29 ) return new PlacedUSolidsTube<10,29>(bp,tm);
	if( tid == 10 && rid == 30 ) return new PlacedUSolidsTube<10,30>(bp,tm);
	if( tid == 10 && rid == 31 ) return new PlacedUSolidsTube<10,31>(bp,tm);
	if( tid == 10 && rid == 32 ) return new PlacedUSolidsTube<10,32>(bp,tm);
	if( tid == 10 && rid == 33 ) return new PlacedUSolidsTube<10,33>(bp,tm);
	if( tid == 10 && rid == 34 ) return new PlacedUSolidsTube<10,34>(bp,tm);
	if( tid == 10 && rid == 35 ) return new PlacedUSolidsTube<10,35>(bp,tm);
	if( tid == 10 && rid == 36 ) return new PlacedUSolidsTube<10,36>(bp,tm);
	if( tid == 10 && rid == 37 ) return new PlacedUSolidsTube<10,37>(bp,tm);
	if( tid == 10 && rid == 38 ) return new PlacedUSolidsTube<10,38>(bp,tm);
	if( tid == 10 && rid == 39 ) return new PlacedUSolidsTube<10,39>(bp,tm);
	if( tid == 10 && rid == 40 ) return new PlacedUSolidsTube<10,40>(bp,tm);
	if( tid == 20 && rid == 20 ) return new PlacedUSolidsTube<20,20>(bp,tm);
	if( tid == 20 && rid == 21 ) return new PlacedUSolidsTube<20,21>(bp,tm);
	if( tid == 20 && rid == 22 ) return new PlacedUSolidsTube<20,22>(bp,tm);
	if( tid == 20 && rid == 23 ) return new PlacedUSolidsTube<20,23>(bp,tm);
	if( tid == 20 && rid == 24 ) return new PlacedUSolidsTube<20,24>(bp,tm);
	if( tid == 20 && rid == 25 ) return new PlacedUSolidsTube<20,25>(bp,tm);
	if( tid == 20 && rid == 26 ) return new PlacedUSolidsTube<20,26>(bp,tm);
	if( tid == 20 && rid == 27 ) return new PlacedUSolidsTube<20,27>(bp,tm);
	if( tid == 20 && rid == 28 ) return new PlacedUSolidsTube<20,28>(bp,tm);
	if( tid == 20 && rid == 29 ) return new PlacedUSolidsTube<20,29>(bp,tm);
	if( tid == 20 && rid == 30 ) return new PlacedUSolidsTube<20,30>(bp,tm);
	if( tid == 20 && rid == 31 ) return new PlacedUSolidsTube<20,31>(bp,tm);
	if( tid == 20 && rid == 32 ) return new PlacedUSolidsTube<20,32>(bp,tm);
	if( tid == 20 && rid == 33 ) return new PlacedUSolidsTube<20,33>(bp,tm);
	if( tid == 20 && rid == 34 ) return new PlacedUSolidsTube<20,34>(bp,tm);
	if( tid == 20 && rid == 35 ) return new PlacedUSolidsTube<20,35>(bp,tm);
	if( tid == 20 && rid == 36 ) return new PlacedUSolidsTube<20,36>(bp,tm);
	if( tid == 20 && rid == 37 ) return new PlacedUSolidsTube<20,37>(bp,tm);
	if( tid == 20 && rid == 38 ) return new PlacedUSolidsTube<20,38>(bp,tm);
	if( tid == 20 && rid == 39 ) return new PlacedUSolidsTube<20,39>(bp,tm);
	if( tid == 20 && rid == 40 ) return new PlacedUSolidsTube<20,40>(bp,tm);
#endif

	// fallback case
	return new PlacedUSolidsTube<1,-1>(bp,tm);
}

