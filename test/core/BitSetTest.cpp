#include "VecGeom/base/BitSet.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include "VecGeom/base/Assert.h"

using namespace vecgeom;

#include <iostream>

template <typename T>
bool compare(size_t iter, T value, T expected)
{
  if (value != expected) {
    std::cerr << "For iter " << iter << " expected " << expected << " and got " << value << "\n";
    return false;
  } else {
    return true;
  }
}

bool CheckValues(BitSet *s, size_t nbits)
{
  VECGEOM_ASSERT(s->CountBits() == 32);

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, s->TestBitNumber(b), (bool)(b % 3))); // every 3rd bit is zero.
  }

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, (*s)[b] == (bool)(b % 3), true)); // every 3rd bit is zero.
  }

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, s->FirstNullBit(b), ((b % 3 == 0) ? b : ((b / 3 + 1) * 3)))); // every 3rd bit is zero.
  }

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, s->FirstSetBit(b), ((b % 3 == 0) ? b + 1 : b))); // every 3rd bit is zero.
  }

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, s->LastNullBit(b), 3 * (b / 3)));
  }

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, s->LastSetBit(b), ((b % 3 == 0) ? (b == 0 ? nbits : b - 1) : b)));
  }

  return true;
}

void BitSetTest()
{
  size_t nbits = 48;

  BitSet *s = BitSet::MakeInstance(nbits);

  // fprintf(stderr,"%ld, %ld vs %ld \n", sizeof(BitSet), s->SizeOf(), (( (nbits+1) )/8 - 1) + sizeof(BitSet));
  VECGEOM_ASSERT(s->SizeOf() == (((nbits + 1)) / 8 - 1) + sizeof(BitSet));

  for (size_t b = 0; b < nbits; ++b) {
    s->SetBitNumber(b, (bool)(b % 3)); // every 3rd bit is zero.
  }

  CheckValues(s, nbits);

  s->ResetAllBits();

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, s->TestBitNumber(b) == false, true)); // All zeros.
  }

  for (size_t b = 0; b < nbits; ++b) {
    s->SetBitNumber(b, true);
  }

  for (size_t b = 0; b < nbits; ++b) {
    VECGEOM_ASSERT(compare(b, s->TestBitNumber(b) == true, true)); // All zeros.
  }

  for (size_t b = 0; b < nbits; ++b) {
    if (!(bool)(b % 3)) {
      s->ResetBitNumber(b); // every 3rd bit is zero.
    }
  }

  CheckValues(s, nbits);
  delete[] s;
}

int main()
{
  BitSetTest();
  return 0;
}
