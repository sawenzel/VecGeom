# --------------------------------------------------------------
# GNUmakefile for SBT USolids test suite
# --------------------------------------------------------------

name := SBT
G4TARGET := $(name)
G4EXLIB := true

ifndef G4INSTALL
  @echo "ERROR - No G4INSTALL defined for Geant4 Toolkit installation path !"
  @exit
endif

CPPFLAGS += -I../include -I../../../include
EXTRALIBS := -lG4USolids ../../../libUSolids.so
EXTRA_LINK_DEPENDENCIES := usolids_libs

.PHONY: all
all: lib bin

include $(G4INSTALL)/config/binmake.gmk

usolids_libs:
	@cd ../../..; $(MAKE)
	@cd ../../../bridges/G4; $(MAKE)
