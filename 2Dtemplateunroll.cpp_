// example of a two loop unrolling operation using templates

#include <cstdio>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

template<int i, int j>
inline
void
dosomething()
{
#pragma message("do something instantiated")
  printf("%d\t %d\n",i,j);
}

template<int upperx, int uppery, const int MAXY>
class EmitCode
{
public:
  static
  void emit()
  {
    dosomething<upperx,uppery>(); 
    EmitCode<upperx, uppery-1, MAXY>::emit();
  }
};


// template specialization
template<const int MAXY>
class EmitCode<0,0,MAXY>
{
public:
  static
  void emit()
  {
    dosomething<0,0>(); 
    return;
  }
};


// template specialization
template<int upperx, const int MAXY>
class EmitCode<upperx,0,MAXY>
{
public:
  static
  void emit()
  {
    dosomething<upperx,0>(); 
    EmitCode<upperx-1, MAXY, MAXY>::emit();
  }
};


// template specialization
template<int uppery, const int MAXY>
class EmitCode<0,uppery,MAXY>
{
public:
  static
  void emit()
  {
    dosomething<0,uppery>(); 
    EmitCode<0, uppery-1, MAXY>::emit();
  }
};



int main()
{
  EmitCode<3,3,3>::emit();
}
