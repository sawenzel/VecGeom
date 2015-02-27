/*
 * XRayBenchmarkFromROOTFile.cpp
 */

#include "VUSolid.hh"
#include "management/RootGeoManager.h"
#include "volumes/LogicalVolume.h"

#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/Stopwatch.h"
#include "navigation/SimpleNavigator.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <map>
#include <cassert>

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoNavigator.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"

#define VERBOSE false   //true or false
#define WRITE_FILE_NAME "volumeImage.bmp" // output image name

using namespace vecgeom;

#pragma pack(push, 1)

typedef struct tFILE_HEADER
{
  unsigned short bfType;
  unsigned long bfSize;
  unsigned short bfReserved1;
  unsigned short bfReserved2;
  unsigned long bfOffBits;
} FILE_HEADER;

#pragma pack(pop)

typedef struct tINFO_HEADER
{
   unsigned long biSize;
   unsigned long biWidth;
   unsigned long biHeight;
   unsigned short biPlanes;
   unsigned short biBitCount;
   unsigned long biCompression;
   unsigned long biSizeImage;
   unsigned long biXPelsPerMeter;
   unsigned long biYPelsPerMeter;
   unsigned long biClrUsed;
   unsigned long biClrImportant;
} INFO_HEADER;

typedef struct tMY_BITMAP
{
  FILE_HEADER  bmpFileHeader;
  INFO_HEADER  bmpInfoHeader;
  unsigned char* bmpPalette;
  unsigned char* bmpRawData;
} MY_BITMAP;

bool usolids= true;

int make_bmp(int* volume_result, int data_size_x, int data_size_y);


//////////////////////////////////
// main function
int main(int argc, char * argv[])
{
  int axis= 0;

  double axis1_start= 0.;
  double axis1_end= 0.;

  double axis2_start= 0.;
  double axis2_end= 0.;

  double pixel_width= 0;
  double pixel_axis= 1.;

  if( argc < 5 )
  {
    std::cerr<< std::endl;
    std::cerr<< "Need to give rootfile, volumename, axis and number of axis"<< std::endl;
    std::cerr<< "USAGE : ./XRayBenchmarkFromROOTFile [rootfile] [VolumeName] [ViewDirection(Axis)] [PixelWidth(OutputImageSize)] [--usolids|--vecgeom(Default:usolids)]"<< std::endl;
    std::cerr<< "  ex) ./XRayBenchmarkFromROOTFile cms2015.root BSCTrap y 95"<< std::endl;
    std::cerr<< "      ./XRayBenchmarkFromROOTFile cms2015.root PLT z 500 --vecgeom"<< std::endl<< std::endl;
    return 1;
  }

  TGeoManager::Import( argv[1] );
  std::string testvolume( argv[2] );

  if( strcmp(argv[3], "x")==0 )
    axis= 1;
  else if( strcmp(argv[3], "y")==0 )
    axis= 2;
  else if( strcmp(argv[3], "z")==0 )
    axis= 3;
  else
  {
    std::cerr<< "Incorrect axis"<< std::endl<< std::endl;
    return 1;
  }

  pixel_width= atof(argv[4]);

  for(auto i= 5; i< argc; i++)
  {
    if( ! strcmp(argv[i], "--usolids") )
      usolids= true;
    
    if( ! strcmp(argv[i], "--vecgeom") )
      usolids= false;
  }

  int found = 0;
  TGeoVolume * foundvolume = NULL;
  // now try to find shape with logical volume name given on the command line
  TObjArray *vlist = gGeoManager->GetListOfVolumes( );
  for( auto i = 0; i < vlist->GetEntries(); ++i )
  {
    TGeoVolume * vol = reinterpret_cast<TGeoVolume*>(vlist->At( i ));
    std::string fullname(vol->GetName());
    
    // strip off pointer information
    std::string strippedname(fullname, 0, fullname.length()-4);
    
    std::size_t founds = strippedname.compare(testvolume);
    if (founds == 0){
      found++;
      foundvolume = vol;

      std::cerr << "("<< i<< ")found matching volume " << foundvolume->GetName()
		<< " of type " << foundvolume->GetShape()->ClassName() << "\n";
    }
  }

  std::cerr << "volume found " << found << " times \n\n";

  if( foundvolume )
  {
    foundvolume->GetShape()->InspectShape();
    std::cerr << "volume capacity " 
	      << foundvolume->GetShape()->Capacity() << "\n";


    // generate benchmark cases
    double dx = ((TGeoBBox*)foundvolume->GetShape())->GetDX()*1.5;
    double dy = ((TGeoBBox*)foundvolume->GetShape())->GetDY()*1.5;
    double dz = ((TGeoBBox*)foundvolume->GetShape())->GetDZ()*1.5;


    double origin[3]= {0., };
    origin[0]= ((TGeoBBox*)foundvolume->GetShape())->GetOrigin()[0];
    origin[1]= ((TGeoBBox*)foundvolume->GetShape())->GetOrigin()[1];
    origin[2]= ((TGeoBBox*)foundvolume->GetShape())->GetOrigin()[2];
    
    TGeoMaterial * matVacuum = new TGeoMaterial("Vacuum",0,0,0);
    TGeoMedium * vac = new TGeoMedium("Vacuum",1,matVacuum);
    TGeoVolume* boundingbox= gGeoManager->MakeBox("BoundingBox", vac, dx, dy, dz);
    
    TGeoManager * geom = boundingbox->GetGeoManager();
    geom->SetTopVolume( boundingbox );
    boundingbox->AddNode( foundvolume, 1, new TGeoTranslation("trans1", 0, 0, 0 ) );
    
    //geom->CloseGeometry();
    //delete world->GetVoxels();
    //world->SetVoxelFinder(0);
    
    TGeoNavigator * nav = geom->GetCurrentNavigator();
    
    std::cout<< std::endl;
    std::cout<< "BoundingBoxDX: "<< dx<< std::endl;
    std::cout<< "BoundingBoxDY: "<< dy<< std::endl;
    std::cout<< "BoundingBoxDZ: "<< dz<< std::endl;
    
    std::cout<< std::endl;
    std::cout<< "BoundingBoxOriginX: "<< origin[0]<< std::endl;
    std::cout<< "BoundingBoxOriginY: "<< origin[1]<< std::endl;
    std::cout<< "BoundingBoxOriginZ: "<< origin[2]<< std::endl<< std::endl;
  
    Vector3D<Precision> p;
    Vector3D<Precision> dir;
    
    if(axis== 1)
    {
      dir.Set(1., 0., 0.);

      axis1_start= origin[1]- dy;
      axis1_end= origin[1]+ dy;

      axis2_start= origin[2]- dz;
      axis2_end= origin[2]+ dz;

      pixel_axis= (dy*2)/pixel_width;
    }
    else if(axis== 2)
    {
      dir.Set(0., 1., 0.);

      axis1_start= origin[0]- dx;
      axis1_end= origin[0]+ dx;

      axis2_start= origin[2]- dz;
      axis2_end= origin[2]+ dz;

      pixel_axis= (dx*2)/pixel_width;
    }
    else if(axis== 3)
    {
      dir.Set(0., 0., 1.);

      axis1_start= origin[0]- dx;
      axis1_end= origin[0]+ dx;

      axis2_start= origin[1]- dy;
      axis2_end= origin[1]+ dy;

      pixel_axis= (dx*2)/pixel_width;
    }
    nav->SetCurrentDirection( dir.x(), dir.y(), dir.z() );


    TGeoNode const * vol;
    double const * cp = nav->GetCurrentPoint();
    double const * cd = nav->GetCurrentDirection();
    int passed_volume= 0;

    int data_size_x= (axis1_end-axis1_start)/pixel_axis;
    int data_size_y= (axis2_end-axis2_start)/pixel_axis;
    int pixel_count_x= 0;
    int pixel_count_y= 0;
    int *volume_result= (int*)malloc((sizeof(int))*data_size_y * data_size_x*3);

    Stopwatch timer;
    timer.Start();
    
    for(double axis2_count= axis2_start; (axis2_count < axis2_end) && (pixel_count_y<data_size_y); axis2_count=axis2_count+pixel_axis)
    {
      for(double axis1_count= axis1_start; (axis1_count < axis1_end) && (pixel_count_x<data_size_x); axis1_count=axis1_count+pixel_axis)
      {
	if(VERBOSE)
	{
	  std::cout << std::endl;
	  std::cout << " OutputPoint("<< axis1_count<< ", "<< axis2_count<< ")"<< std::endl;
	}
	
	if( axis== 1)
	  p.Set( origin[0]-dx, axis1_count, axis2_count);
	else if( axis== 2)
	  p.Set( axis1_count, origin[1]-dy, axis2_count);
	else if( axis== 3)
	  p.Set( axis1_count, axis2_count, origin[2]-dz);
	
	nav->SetCurrentPoint( p.x(), p.y(), p.z() );
	
	double distancetravelled=0.;
	passed_volume= 0;
	
	if(VERBOSE)
	{
	  std::cout << " StartPoint(" << cp[0] << ", " << cp[1] << ", " << cp[2] << ")";
	  std::cout << " Direction <" << cd[0] << ", " << cd[1] << ", " << cd[2] << ">"<< std::endl;
	}
	
	
	while( vol = nav->FindNextBoundaryAndStep( kInfinity ) )
        {
	  distancetravelled+=nav->GetStep();
	  
	  if(VERBOSE)
	  {
	    if( vol != NULL )
	      std::cout << "  VolumeName: "<< vol->GetVolume()->GetName();
	    else
	      std::cout << "  NULL: ";
	    std::cout << " point(" << cp[0] << ", " << cp[1] << ", " << cp[2] << ")";
	    std::cout << " step[" << nav->GetStep()<< "]"<< std::endl;
	  }
	  
	  // Increase passed_volume
	  passed_volume++;
	  
	  cp = nav->GetCurrentPoint();
	  cd = nav->GetCurrentDirection();
	  
	}
	
	///////////////////////////////////
	// Store the number of passed volume at 'volume_result'
	*(volume_result+pixel_count_y*data_size_x+pixel_count_x)= passed_volume;

	if(VERBOSE)
	{
	  std::cout << "  EndOfBoundingBox:";
	  std::cout << " point(" << cp[0] << ", " << cp[1] << ", " << cp[2] << ")";
	  std::cout << " PassedVolume:" << "<"<< passed_volume<< "("<< passed_volume%10<< ")>";
	  std::cout << " step[" << nav->GetStep()<< "]";
	  std::cout << " Distance: " << distancetravelled<< std::endl;
	}
	pixel_count_x++;
      }

      pixel_count_x= 0;
      pixel_count_y++;
    }
   
    timer.Stop();

    std::cout << std::endl;
    std::cout << " Elapsed time : "<< timer.Elapsed() << std::endl;

    // Make bitmap file
    make_bmp(volume_result, data_size_x, data_size_y);
    std::cout << " Result is stored at 'volumeImage.bmp'"<< std::endl<< std::endl;

    delete volume_result;
  }
  return 0;
}


int make_bmp(int* volume_result, int data_size_x, int data_size_y)
{

  MY_BITMAP* pBitmap= new MY_BITMAP;
  FILE *pBitmapFile;
  int width_4= (data_size_x+ 3)&~3;
  unsigned char* bmpBuf;

  bmpBuf = (unsigned char*)malloc(sizeof(unsigned char)* (data_size_y* width_4* 3+ 54));
  printf("\n Write bitmap...\n");

  unsigned int len= 0;

  // bitmap file header
  pBitmap->bmpFileHeader.bfType=0x4d42;
  pBitmap->bmpFileHeader.bfSize=data_size_y* width_4* 3+ 54;
  pBitmap->bmpFileHeader.bfReserved1= 0;
  pBitmap->bmpFileHeader.bfReserved2= 0;
  pBitmap->bmpFileHeader.bfOffBits= 54;
  
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfType, 2);
  len+= 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfSize, 4);
  len+= 4;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved1, 2);
  len+= 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfReserved2, 2);
  len+= 2;
  memcpy(bmpBuf + len, &pBitmap->bmpFileHeader.bfOffBits, 4);
  len+= 4;

  // bitmap information header
  pBitmap->bmpInfoHeader.biSize= 40;
  pBitmap->bmpInfoHeader.biWidth= width_4;
  pBitmap->bmpInfoHeader.biHeight= data_size_y;
  pBitmap->bmpInfoHeader.biPlanes= 1;
  pBitmap->bmpInfoHeader.biBitCount= 24;
  pBitmap->bmpInfoHeader.biCompression= 0;
  pBitmap->bmpInfoHeader.biSizeImage= data_size_y* width_4* 3;
  pBitmap->bmpInfoHeader.biXPelsPerMeter= 0;
  pBitmap->bmpInfoHeader.biYPelsPerMeter= 0;
  pBitmap->bmpInfoHeader.biClrUsed= 0;
  pBitmap->bmpInfoHeader.biClrImportant=0;


  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biSize, 4); 
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biWidth, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biHeight, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biPlanes, 2);
  len+= 2;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biBitCount, 2);
  len+= 2;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biCompression, 4); 
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biSizeImage, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biXPelsPerMeter, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biYPelsPerMeter, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biClrUsed, 4);
  len+= 4;
  memcpy(bmpBuf+len, &pBitmap->bmpInfoHeader.biClrImportant, 4);
  len+= 4;


  int x= 0;
  int y= 0;
  int origin_x= 0;

  int padding= width_4- data_size_x;
  int padding_idx= padding;
  unsigned char *imgdata= (unsigned char*)malloc(sizeof(unsigned char)*data_size_y*width_4*3);

  while( y< data_size_y )
  {
    while( origin_x< data_size_x )
    {
      *(imgdata+y*width_4*3+x*3+0)= (*(volume_result+y*data_size_x+origin_x) *50) %256;
      *(imgdata+y*width_4*3+x*3+1)= (*(volume_result+y*data_size_x+origin_x) *40) %256;
      *(imgdata+y*width_4*3+x*3+2)= (*(volume_result+y*data_size_x+origin_x) *30) %256;
      
      x++;
      origin_x++;

      while( origin_x== data_size_x && padding_idx)
      {
	// padding 4-byte at bitmap image
	*(imgdata+y*width_4*3+x*3+0)= 0;
	*(imgdata+y*width_4*3+x*3+1)= 0;
	*(imgdata+y*width_4*3+x*3+2)= 0;
	x++;
	padding_idx--;

      }
      padding_idx= padding;
    }
    y++;
    x= 0;
    origin_x= 0;
  }
  
  memcpy(bmpBuf + 54, imgdata, width_4* data_size_y* 3);

  pBitmapFile = fopen(WRITE_FILE_NAME, "wb");
  fwrite(bmpBuf, sizeof(char), width_4*data_size_y*3+54, pBitmapFile);


  fclose(pBitmapFile);
  delete imgdata;
  delete pBitmap;
  delete bmpBuf;

  return 0;
}
