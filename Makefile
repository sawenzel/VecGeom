SrcSuf        = cc
ObjSuf        = o
DllSuf        = so
OutPutOpt     = -o # keep whitespace after "-o"

OPT           = -g

CXX           = g++
CXXFLAGS      = $(OPT) -Wall -fPIC
LD            = g++
LDFLAGS       = $(OPT)
SOFLAGS       = -shared -Wl -m64 -g
INCDIR        = include
INCBRIDGE     = bridges/TGeo
SRCDIR        = src
CXXFLAGS     += -I./$(INCDIR)


SOURCES       = UUtils.cc UVector3.cc UTransform3D.cc VUSolid.cc UBox.cc UMultiUnion.cc UVoxelFinder.cc
UBRIDGETGEO   = TGeoUShape.cxx
UBRIDGEDICTS  = G__UBridges.cxx
UBRIDGETGEOO  = TGeoUShape.o G__UBridges.o
UBRIDGESL     = $(patsubst %.o,$(INCBRIDGE)/%.o,$(UBRIDGETGEOO))
USOLIDSS      = $(patsubst %,$(SRCDIR)/%,$(SOURCES))
USOLIDSO      = $(patsubst %.cc,%.o,$(SOURCES))
USOLIDSL      = $(patsubst %.cc,$(SRCDIR)/%.o,$(SOURCES))
USOLIDDEPS    = $(patsubst %.cc,$(INCDIR)/%.hh,$(SOURCES))

USOLIDSSO     = libUSolids.$(DllSuf)
UBRIDGESSO    = libUBridges.$(DllSuf)
USOLIDSLIB    = $(shell pwd)/$(USOLIDSSO)
OBJS          = $(USOLIDSO)


#------------------------------------------------------------------------------

all:            $(USOLIDSSO)
bridges:        $(UBRIDGESSO)

$(USOLIDSSO):     $(USOLIDSO)
		$(LD) $(LDFLAGS) $(SOFLAGS) $(USOLIDSL) $(OutPutOpt)$@
		@echo "$@ done"
$(UBRIDGESSO):    $(UBRIDGEDICTS) $(UBRIDGETGEOO)
		$(LD) $(LDFLAGS) $(SOFLAGS) $(UBRIDGESL) $(OutPutOpt)$@
		@echo "$@ done"
UUtils.o: src/UUtils.cc
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/UUtils.cc
UVector3.o:
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/UVector3.cc
UTransform3D.o:
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/UTransform3D.cc
VUSolid.o:
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/VUSolid.cc
UBox.o:
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/UBox.cc   
UMultiUnion.o:
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/UMultiUnion.cc    
UVoxelFinder.o:
		$(CXX)  $(CXXFLAGS) -o src/$@ -c src/UVoxelFinder.cc                   
TGeoUShape.o:
		$(CXX)  $(CXXFLAGS) -I$(ROOTSYS)/include -o $(INCBRIDGE)/$@ -c $(INCBRIDGE)/$(UBRIDGETGEO)
$(UBRIDGEDICTS):
		@echo "Generating dictionary $@"
#		cp $(INCBRIDGE)/*.h $(INCDIR)
		$(ROOTSYS)/bin/rootcint -f $(INCBRIDGE)/$@ \
                -c $(INCBRIDGE)/TGeoUShape.h $(INCBRIDGE)/LinkDef.h
G__UBridges.o:
		$(ROOTSYS)/bin/rmkdepend -R -f$(INCBRIDGE)/G__UBridges.d -Y -w 1000 -- \
                pipe -m64 -Wshadow -Wall -W -Woverloaded-virtual -fPIC \
                -I$(INCBRIDGE) -pthread  -D__cplusplus -I$(ROOTSYS)cint/cint/lib/prec_stl \
                -I$(ROOTSYS)cint/cint/stl -I$(ROOTSYS)/cint/cint/inc -- $(INCBRIDGE)/$(UBRIDGEDICTS)
		$(CXX)  $(CXXFLAGS) -I. -I$(ROOTSYS)/include -o $(INCBRIDGE)/$@ \
                -c $(INCBRIDGE)/G__UBridges.cxx
clean:
		@rm -rf $(USOLIDSSO) src/*.o
                

