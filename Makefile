SrcSuf        = cc
ObjSuf        = o
DllSuf        = so
OutPutOpt     = -o # keep whitespace after "-o"

OPT           = -g

CXX           = g++
CXXFLAGS      = $(OPT) -Wall -fPIC
LD            = g++
LDFLAGS       = $(OPT)
SOFLAGS       = -shared
INCDIR        = include
SRCDIR        = src
CXXFLAGS     += -I./$(INCDIR)


SOURCES       = UUtils.cc UVector3.cc VUSolid.cc UBox.cc
USOLIDSS      = $(patsubst %,$(SRCDIR)/%,$(SOURCES))
USOLIDSO      = $(patsubst %.cc,%.o,$(SOURCES))
USOLIDSL      = $(patsubst %.cc,$(SRCDIR)/%.o,$(SOURCES))
USOLIDDEPS    = $(patsubst %.cc,$(INCDIR)/%.hh,$(SOURCES))

USOLIDSSO     = libUSolids.$(DllSuf)
USOLIDSLIB    = $(shell pwd)/$(USOLIDSSO)
OBJS          = $(USOLIDSO)


#------------------------------------------------------------------------------

all:            $(USOLIDSSO)


$(USOLIDSSO):     $(USOLIDSO)
		$(LD) $(LDFLAGS) $(SOFLAGS) $(USOLIDSL) $(OutPutOpt)$@
		@echo "$@ done"

UUtils.o: src/UUtils.cc
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/UUtils.cc
UVector3.o:
		$(CXX)  $(CXXFLAGS) -o src/UVector3.o -c src/UVector3.cc
VUSolid.o:
		$(CXX)  $(CXXFLAGS) -o src/VUSolid.o -c src/VUSolid.cc
UBox.o:
		$(CXX)  $(CXXFLAGS) -o src/UBox.o -c src/UBox.cc
                
clean:
		@rm -rf $(USOLIDSSO) src/*.o
                

