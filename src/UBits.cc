
// Author: Philippe Canal 05/02/2001
//    Feb  5 2001: Creation
//    Feb  6 2001: Changed all int to unsigned int.

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// UBits                                                                //
//                                                                      //
// Container of bits                                                    //
//                                                                      //
// This class provides a simple container of bits.                      //
// Each bit can be set and tested via the functions SetBitNumber and    //
// TestBitNumber.                                             .         //
// The default value of all bits is false.                             //
// The size of the container is automatically extended when a bit       //
// number is either set or tested.  To reduce the memory size of the    //
// container use the Compact function, this will discard the memory     //
// occupied by the upper bits that are 0.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "UBits.h"
#include "string.h"

//#include "Ristd::ostream.h"

//ClassImp(UBits)

//______________________________________________________________________________
UBits::UBits(unsigned int nBits) : nBits(nBits)
{
   // UBits constructor.  All bits set to 0

   if (nBits <= 0) nBits = 8;
   nBytes  = ((nBits-1)/8) + 1;
   allBits = new unsigned char[nBytes];
   // this is redundant only with libNew
   memset(allBits,0,nBytes);
}

//______________________________________________________________________________
UBits::UBits(const UBits &original) : nBits(original.nBits),
   nBytes(original.nBytes)
{
   // UBits copy constructor

   allBits = new unsigned char[nBytes];
   memcpy(allBits,original.allBits,nBytes);

}

//______________________________________________________________________________
UBits& UBits::operator=(const UBits& rhs)
{
   // UBits assignment operator

   if (this != &rhs) {
//      TObject::operator=(rhs);
      nBits   = rhs.nBits;
      nBytes  = rhs.nBytes;
      delete [] allBits;
      if (nBytes != 0) {
         allBits = new unsigned char[nBytes];
         memcpy(allBits,rhs.allBits,nBytes);
      } else {
         allBits = 0;
      }
   }
   return *this;
}

//______________________________________________________________________________
UBits::~UBits()
{
   // UBits destructor

   delete [] allBits;
}

//______________________________________________________________________________
void UBits::Clear()
{
   // Clear the value.

   delete [] allBits;
   allBits = 0;
   nBits   = 0;
   nBytes  = 0;
}

//______________________________________________________________________________
void UBits::Compact()
{
   // Reduce the storage used by the object to a minimun

   if (!nBits || !allBits) return;
   unsigned int needed;
   for(needed=nBytes-1;
       needed > 0 && allBits[needed]==0; ) { needed--; };
   needed++;

   if (needed!=nBytes) {
      unsigned char *old_location = allBits;
      allBits = new unsigned char[needed];

      memcpy(allBits,old_location,needed);
      delete [] old_location;

      nBytes = needed;
      nBits = 8*nBytes;
   }
}

//______________________________________________________________________________
unsigned int UBits::CounUBits(unsigned int startBit) const
{
   // Return number of bits set to 1 starting at bit startBit

   static const int nBitsCached[256] = {
             0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

   unsigned int i,count = 0;
   if (startBit == 0) {
      for(i=0; i<nBytes; i++) {
         count += nBitsCached[allBits[i]];
      }
      return count;
   }
   if (startBit >= nBits) return count;
   unsigned int startByte = startBit/8;
   unsigned int ibit = startBit%8;
   if (ibit) {
      for (i=ibit;i<8;i++) {
         if (allBits[startByte] & (1<<ibit)) count++;
      }
      startByte++;
   }
   for(i=startByte; i<nBytes; i++) {
      count += nBitsCached[allBits[i]];
   }
   return count;
}

//______________________________________________________________________________
void UBits::DoAndEqual(const UBits& rhs)
{
   // Execute (*this) &= rhs;
   // Extra bits in rhs are ignored
   // Missing bits in rhs are assumed to be zero.

   unsigned int min = (nBytes<rhs.nBytes) ? nBytes : rhs.nBytes;
   for(unsigned int i=0; i<min; ++i) {
      allBits[i] &= rhs.allBits[i];
   }
   if (nBytes>min) {
      memset(&(allBits[min]),0,nBytes-min);
   }
}

//______________________________________________________________________________
void UBits::DoOrEqual(const UBits& rhs)
{
   // Execute (*this) &= rhs;
   // Extra bits in rhs are ignored
   // Missing bits in rhs are assumed to be zero.

   unsigned int min = (nBytes<rhs.nBytes) ? nBytes : rhs.nBytes;
   for(unsigned int i=0; i<min; ++i) {
      allBits[i] |= rhs.allBits[i];
   }
}

//______________________________________________________________________________
void UBits::DoXorEqual(const UBits& rhs)
{
   // Execute (*this) ^= rhs;
   // Extra bits in rhs are ignored
   // Missing bits in rhs are assumed to be zero.

   unsigned int min = (nBytes<rhs.nBytes) ? nBytes : rhs.nBytes;
   for(unsigned int i=0; i<min; ++i) {
      allBits[i] ^= rhs.allBits[i];
   }
}

//______________________________________________________________________________
void UBits::DoFlip()
{
   // Execute ~(*this)

   for(unsigned int i=0; i<nBytes; ++i) {
      allBits[i] = ~allBits[i];
   }
   // NOTE: out-of-bounds bit were also flipped!
}

//______________________________________________________________________________
void UBits::DoLeftShift(unsigned int shift)
{
   // Execute the left shift operation.

   if (shift==0) return;
   const unsigned int wordshift = shift / 8;
   const unsigned int offset = shift % 8;
   if (offset==0) {
      for(unsigned int n = nBytes - 1; n >= wordshift; --n) {
         allBits[n] = allBits[ n - wordshift ];
      }
   } else {
      const unsigned int sub_offset = 8 - offset;
      for(unsigned int n = nBytes - 1; n > wordshift; --n) {
         allBits[n] = (allBits[n - wordshift] << offset) |
                       (allBits[n - wordshift - 1] >> sub_offset);
      }
      allBits[wordshift] = allBits[0] << offset;
   }
   memset(allBits,0,wordshift);
}

//______________________________________________________________________________
void UBits::DoRightShift(unsigned int shift)
{
   // Execute the left shift operation.

   if (shift==0) return;
   const unsigned int wordshift = shift / 8;
   const unsigned int offset = shift % 8;
   const unsigned int limit = nBytes - wordshift - 1;

   if (offset == 0)
      for (unsigned int n = 0; n <= limit; ++n)
         allBits[n] = allBits[n + wordshift];
   else
   {
      const unsigned int sub_offset = 8 - offset;
      for (unsigned int n = 0; n < limit; ++n)
         allBits[n] = (allBits[n + wordshift] >> offset) |
                       (allBits[n + wordshift + 1] << sub_offset);
      allBits[limit] = allBits[nBytes-1] >> offset;
   }

   memset(&(allBits[limit + 1]),0, nBytes - limit - 1);
}

//______________________________________________________________________________
unsigned int UBits::FirstNullBit(unsigned int startBit) const
{
   // Return position of first null bit (starting from position 0 and up)

   static const int fbits[256] = {
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,6,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,7,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,6,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,8};

   unsigned int i;
   if (startBit == 0) {
      for(i=0; i<nBytes; i++) {
         if (allBits[i] != 255) return 8*i + fbits[allBits[i]];
      }
      return nBits;
   }
   if (startBit >= nBits) return nBits;
   unsigned int startByte = startBit/8;
   unsigned int ibit = startBit%8;
   if (ibit) {
      for (i=ibit;i<8;i++) {
         if ((allBits[startByte] & (1<<i)) == 0) return 8*startByte+i;
      }
      startByte++;
   }
   for(i=startByte; i<nBytes; i++) {
      if (allBits[i] != 255) return 8*i + fbits[allBits[i]];
   }
   return nBits;
}

//______________________________________________________________________________
unsigned int UBits::FirstSetBit(unsigned int startBit) const
{
   // Return position of first non null bit (starting from position 0 and up)

   static const int fbits[256] = {
             8,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             7,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0};

   unsigned int i;
   if (startBit == 0) {
      for(i=0; i<nBytes; i++) {
         if (allBits[i] != 0) return 8*i + fbits[allBits[i]];
      }
      return nBits;
   }
   if (startBit >= nBits) return nBits;
   unsigned int startByte = startBit/8;
   unsigned int ibit = startBit%8;
   if (ibit) {
      for (i=ibit;i<8;i++) {
         if ((allBits[startByte] & (1<<i)) != 0) return 8*startByte+i;
      }
      startByte++;
   }
   for(i=startByte; i<nBytes; i++) {
      if (allBits[i] != 0) return 8*i + fbits[allBits[i]];
   }
   return nBits;
}

//______________________________________________________________________________
void UBits::Output(std::ostream &os) const
{
   // Print the value to the std::ostream

   for(unsigned int i=0; i<nBytes; ++i) {
      unsigned char val = allBits[nBytes - 1 - i];
      for (unsigned int j=0; j<8; ++j) {
         os << (bool)(val&0x80);
         val <<= 1;
      }
   }
}

//______________________________________________________________________________
void UBits::Paint()
{
   // Once implemented, it will draw the bit field as an histogram.
   // use the TVirtualPainter as the usual trick

}

//______________________________________________________________________________
void UBits::Print() const
{
   // Print the list of active bits

   int count = 0;
   for(unsigned int i=0; i<nBytes; ++i) {
      unsigned char val = allBits[i];
      for (unsigned int j=0; j<8; ++j) {
         if (val & 1) printf(" bit:%4d = 1\n",count);
         count++;
         val = val >> 1;
      }
   }
}

//______________________________________________________________________________
void UBits::ResetAllBits(bool value)
{
   // Reset all bits to 0 (false).

   if (allBits) memset(allBits, value ? 0xFF : 0,nBytes);
}

//______________________________________________________________________________
void UBits::ReserveBytes(unsigned int nbytes)
{
   // Reverse each bytes.

   if (nbytes > nBytes) {
      // do it in this order to remain exception-safe.
      unsigned char *newBits=new unsigned char[nbytes];
      delete[] allBits;
      nBytes=nbytes;
      allBits=newBits;
   }
}

//______________________________________________________________________________
void UBits::Set(unsigned int nBits, const char *array)
{
   // Set all the bytes.

   unsigned int nbytes=(nBits+7)>>3;

   ReserveBytes(nbytes);

   nBits=nBits;
   memcpy(allBits, array, nbytes);
}

//______________________________________________________________________________
void UBits::Get(char *array) const
{
   // Copy all the byes.

   memcpy(array, allBits, (nBits+7)>>3);
}

 #ifdef R__BYTESWAP  /* means we are on little endian */

 /*
 If we are on a little endian machine, a bitvector represented using
 any integer type is identical to a bitvector represented using bytes.
 -- FP.
 */

void UBits::Set(unsigned int nBits, const short *array)
{
   // Set all the bytes.

   Set(nBits, (const char*)array);
}

void UBits::Set(unsigned int nBits, const int *array)
{
   // Set all the bytes.

   Set(nBits, (const char*)array);
}

void UBits::Set(unsigned int nBits, const Long64_t *array)
{
   // Set all the bytes.

   Set(nBits, (const char*)array);
}

void UBits::Get(short *array) const
{
   // Get all the bytes.

   Get((char*)array);
}

void UBits::Get(int *array) const
{
   // Get all the bytes.

   Get((char*)array);
}

void UBits::Get(Long64_t *array) const
{
   // Get all the bytes.

   Get((char*)array);
}

#else

 /*
 If we are on a big endian machine, some swapping around is required.
 */

void UBits::Set(unsigned int nBits, const short *array)
{
   // make nbytes even so that the loop below is neat.
   unsigned int nbytes = ((nBits+15)>>3)&~1;

   ReserveBytes(nbytes);

   nBits=nBits;

   const unsigned char *cArray = (const unsigned char*)array;
   for (unsigned int i=0; i<nbytes; i+=2) {
      allBits[i] = cArray[i+1];
      allBits[i+1] = cArray[i];
   }
}

void UBits::Set(unsigned int nBits, const int *array)
{
   // make nbytes a multiple of 4 so that the loop below is neat.
   unsigned int nbytes = ((nBits+31)>>3)&~3;

   ReserveBytes(nbytes);

   nBits=nBits;

   const unsigned char *cArray = (const unsigned char*)array;
   for (unsigned int i=0; i<nbytes; i+=4) {
      allBits[i] = cArray[i+3];
      allBits[i+1] = cArray[i+2];
      allBits[i+2] = cArray[i+1];
      allBits[i+3] = cArray[i];
   }
}

void UBits::Set(unsigned int nBits, const Long64_t *array)
{
   // make nbytes a multiple of 8 so that the loop below is neat.
   unsigned int nbytes = ((nBits+63)>>3)&~7;

   ReserveBytes(nbytes);

   nBits=nBits;

   const unsigned char *cArray = (const unsigned char*)array;
   for (unsigned int i=0; i<nbytes; i+=8) {
      allBits[i] = cArray[i+7];
      allBits[i+1] = cArray[i+6];
      allBits[i+2] = cArray[i+5];
      allBits[i+3] = cArray[i+4];
      allBits[i+4] = cArray[i+3];
      allBits[i+5] = cArray[i+2];
      allBits[i+6] = cArray[i+1];
      allBits[i+7] = cArray[i];
   }
}

void UBits::Get(short *array) const
{
   // Get all the bytes.

   unsigned int nBytes = (nBits+7)>>3;
   unsigned int nSafeBytes = nBytes&~1;

   unsigned char *cArray=(unsigned char*)array;
   for (unsigned int i=0; i<nSafeBytes; i+=2) {
      cArray[i] = allBits[i+1];
      cArray[i+1] = allBits[i];
   }

   if (nBytes>nSafeBytes) {
      cArray[nSafeBytes+1] = allBits[nSafeBytes];
   }
}

void UBits::Get(int *array) const
{
   // Get all the bytes.

   unsigned int nBytes = (nBits+7)>>3;
   unsigned int nSafeBytes = nBytes&~3;

   unsigned char *cArray=(unsigned char*)array;
   unsigned int i;
   for (i=0; i<nSafeBytes; i+=4) {
      cArray[i] = allBits[i+3];
      cArray[i+1] = allBits[i+2];
      cArray[i+2] = allBits[i+1];
      cArray[i+3] = allBits[i];
   }

   for (i=0; i<nBytes-nSafeBytes; ++i) {
      cArray[nSafeBytes + (3 - i)] = allBits[nSafeBytes + i];
   }
}

void UBits::Get(Long64_t *array) const
{
   // Get all the bytes.

   unsigned int nBytes = (nBits+7)>>3;
   unsigned int nSafeBytes = nBytes&~7;

   unsigned char *cArray=(unsigned char*)array;
   unsigned int i;
   for (i=0; i<nSafeBytes; i+=8) {
      cArray[i] = allBits[i+7];
      cArray[i+1] = allBits[i+6];
      cArray[i+2] = allBits[i+5];
      cArray[i+3] = allBits[i+4];
      cArray[i+4] = allBits[i+3];
      cArray[i+5] = allBits[i+2];
      cArray[i+6] = allBits[i+1];
      cArray[i+7] = allBits[i];
   }

   for (i=0; i<nBytes-nSafeBytes; ++i) {
      cArray[nSafeBytes + (7 - i)] = allBits[nSafeBytes + i];
   }
}

#endif

bool UBits::operator==(const UBits &other) const
{
   // Compare object.

   if (nBits == other.nBits) {
      return !memcmp(allBits, other.allBits, (nBits+7)>>3);
   } else if (nBits <  other.nBits) {
      return !memcmp(allBits, other.allBits, (nBits+7)>>3) && other.FirstSetBit(nBits) == other.nBits;
   } else {
      return !memcmp(allBits, other.allBits, (other.nBits+7)>>3) && FirstSetBit(other.nBits) == nBits;
   }
}

