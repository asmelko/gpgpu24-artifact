HOST=$(shell hostname -s)
ifeq ($(HOST), grafik)
ARCH=52
else
ARCH=70
endif

CPP=g++
STD=-std=c++17
CFLAGS=-Wall -O3 $(STD) -DNDEBUG
NVCCFLAGS=-ccbin $(CPP) $(STD) -arch=sm_$(ARCH)
INCLUDE=/usr/local/cuda/include ./headers ./kernels ../../bpplib/include
LDFLAGS=
LIBS=
LIBDIRS=/usr/local/cuda/lib64
HEADERS=$(shell find . -name '*.hpp') $(shell find . -name '*.h') $(shell find . -name '*.cuh')
OBJS=$(patsubst %.cpp,%.obj,$(shell find . -name '*.cpp'))
CUOBJS=$(patsubst %.cu,%.cuobj,$(shell find . -name '*.cu'))
TARGET=esom-$(HOST)
#OTHER_CPP=kmcuda_adapter.cpp $(shell find ./kmcuda '*.cc')

.PHONY: all clear clean purge

all: $(TARGET)



# Building Targets

$(TARGET): $(HEADERS) $(CUOBJS) $(OBJS)
	@echo Linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) -lcudart $(CUOBJS) $(OBJS) -o $@

%.obj: %.cpp $(HEADERS)
	@echo Compiling CPP file $< ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) -c $< -o $@

%.cuobj: %.cu $(HEADERS)
	@echo Compiling kernels in $< ...
	@nvcc $(NVCCFLAGS) $(addprefix -I,$(INCLUDE)) --compile -cudart static $< -o $@


# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f */*.cuobj
	-@rm -f */*.o
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGET)
