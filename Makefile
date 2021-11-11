ifneq ($(wildcard /usr/local/cuda-8.0/.*),)
cxx_builtin_include_directory="/usr/local/cuda-8.0/include"
cuda_library_dir="/usr/local/cuda-8.0/nvvm/libdevice"
else
cxx_builtin_include_directory="/usr/lib/nvidia-cuda-toolkit/include"
cuda_library_dir="/usr/lib/nvidia-cuda-toolkit/libdevice"
endif

ifeq ($(origin JAVA_HOME),undefined)
JAVA_HOME:=/usr/lib/jvm/java-11-openjdk-amd64/include
endif
ifeq ("$(wildcard $(JAVA_HOME)/jni.h)","")
JAVA_HOME:=/usr/lib/jvm/java-8-openjdk-amd64/include
ifeq ("$(wildcard $(JAVA_HOME)/jni.h)","")
JAVA_HOME:=/usr/lib/jvm/java-14-openjdk-amd64/include
ifeq ("$(wildcard $(JAVA_HOME)/jni.h)","")
JAVA_HOME:=/usr/lib/jvm/adoptopenjdk-8-hotspot-amd64/include
ifeq ("$(wildcard $(JAVA_HOME)/jni.h)","")
JAVA_HOME:=/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/include
ifeq ("$(wildcard $(JAVA_HOME)/jni.h)","")
JAVA_HOME:=/opt/hostedtoolcache/Java_Adopt_jdk/11.0.12-7/x64/include
endif
endif
endif
endif
endif
#-I"${JAVA_HOME}/include" -I"${JAVA_HOME}/include/linux"
# -maxrregcount 36

PT_BIN := $(shell python3-config '--extension-suffix')
CFLAGS= -Wall -Wextra -pedantic -g -O2 -fopenmp -std=c++14 -lstdc++fs -mavx2 -ljpeg -lpng -fPIC
LFLAGS= -lstdc++fs -lpng -ljpeg
CFLAGS += -DNDEBUG

ifeq ($(origin NCUDA),undefined)
LIBS:=
else
LIBS:= -lcudart  -L/usr/local/cuda-8.0/lib64/
endif
SRC := src
BUILD := build
G++ := g++-7
CGCC := gcc-8
CC := g++

#debug: CCFLAGS += -DDEBUG -g

CFILES:=util.cpp image_io.cpp serialize.cpp image_util.cpp io_util.cpp raytrace_test.cpp
OFILES:=$(foreach f,$(CFILES),$(subst .cpp,.o,$f))

define built_object
$(BUILD)/$(subst .cpp,.o,$f): $(SRC)/$f $(SRC)/$(subst .cpp,.h,$f)
	$(CC) -c $(CFLAGS) $(DFLAGS) $(LDFLAGS) $(SRC)/$f -o $(BUILD)/$(subst .cpp,.o,$f)

cuda_test: $(BUILD)/$(subst .cpp,.o,$f)
cuda_unit_test: $(BUILD)/$(subst .cpp,.o,$f)
cuda_raytrace_java.so: $(BUILD)/$(subst .cpp,.o,$f)
endef
$(foreach f,$(CFILES),$(eval $(call built_object)))

$(BUILD)/test_main.o: $(SRC)/test_main.cpp $(SRC)/test_main.h $(SRC)/serialize_test.h
	$(CC) -c $(CFLAGS) $(DFLAGS) $(LDFLAGS) $(SRC)/test_main.cpp -o $(BUILD)/test_main.o

ifeq ($(origin NCUDA),undefined)
$(BUILD)/raytracer.o: $(SRC)/cuda_volume_raytracer.cpp $(SRC)/cuda_volume_raytracer.h $(SRC)/tuple_math.h
	$(CC) -I$(cxx_builtin_include_directory) -D_FORCE_INLINES -O2 -v -c $(SRC)/cuda_volume_raytracer.cpp -o $(BUILD)/raytracer.o -ldir=$(cuda_library_dir) -fPIC -g -std=c++11 -fopenmp -msse -msse2 -DNDEBUG -mavx2 -DNCUDA
else
$(BUILD)/raytracer.o: $(SRC)/cuda_volume_raytracer.cpp $(SRC)/cuda_volume_raytracer.h $(SRC)/tuple_math.h
	nvcc -ccbin $(CGCC) -I$(cxx_builtin_include_directory) -D_FORCE_INLINES -O2 -v -c $(SRC)/cuda_volume_raytracer.cpp -o $(BUILD)/raytracer.o --dont-use-profile -ldir=$(cuda_library_dir) --ptxas-options=-v -Xcompiler -fPIC -g -std=c++11 -Xcompiler -fopenmp -Xcompiler -msse -Xcompiler -msse2 -DNDEBUG -Xcompiler -mavx2
endif

$(BUILD)/python_binding.o: $(SRC)/python_binding.cpp
	$(G++) $(SRC)/python_binding.cpp -c $(CFLAGS) -o $(BUILD)/python_binding.o -fPIC `python3 -m pybind11 --includes`

$(BUILD)/java_binding.o: $(SRC)/java_binding.cpp $(SRC)/java_binding.h
	$(CC) $(SRC)/java_binding.cpp -c $(CFLAGS) -o $(BUILD)/java_binding.o -fPIC -I$(jni_library) -I$(jni_library)/linux

cuda_raytrace$(PT_BIN): $(BUILD)/image_io.o $(BUILD)/image_util.o $(BUILD)/raytracer.o $(BUILD)/serialize.o $(BUILD)/python_binding.o $(BUILD)/io_util.o $(BUILD)/serialize.o $(BUILD)/util.o
	$(G++) $(CFLAGS) $(BUILD)/image_io.o $(BUILD)/image_util.o $(BUILD)/raytracer.o $(BUILD)/serialize.o $(BUILD)/util.o $(BUILD)/io_util.o $(BUILD)/python_binding.o $(LIBS) -shared -fPIC -o cuda_raytrace$(PT_BIN) -I/usr/include/python3.6m/ $(LFLAGS)

cuda_raytrace_java.so:  $(BUILD)/raytracer.o $(BUILD)/java_binding.o
	$(CC) $(CFLAGS) -lc $(BUILD)/image_io.o $(BUILD)/image_util.o $(BUILD)/raytracer.o $(BUILD)/util.o $(BUILD)/io_util.o $(BUILD)/serialize.o $(BUILD)/java_binding.o $(LIBS) -shared -fPIC -o cuda_raytrace_java.so -I$(jni_library) -I$(jni_library)/linux $(LFLAGS)

cuda_test: $(BUILD)/image_util.o $(BUILD)/image_io.o $(BUILD)/raytracer.o $(BUILD)/raytrace_test.o $(BUILD)/io_util.o $(BUILD)/serialize.o $(BUILD)/util.o
	$(CC) $(CFLAGS) $(BUILD)/image_util.o $(BUILD)/image_io.o $(BUILD)/raytracer.o $(BUILD)/serialize.o $(BUILD)/util.o $(BUILD)/io_util.o $(BUILD)/raytrace_test.o $(LIBS) -fPIC -o cuda_test $(LFLAGS)

cuda_unit_test: $(BUILD)/raytracer.o $(BUILD)/test_main.o
	$(CC) $(CFLAGS) $(BUILD)/image_util.o $(BUILD)/image_io.o $(BUILD)/raytracer.o $(BUILD)/serialize.o $(BUILD)/util.o $(BUILD)/io_util.o $(BUILD)/test_main.o $(LIBS) -fPIC -lboost_unit_test_framework -no-pie $(LFLAGS) -o cuda_unit_test

test: cuda_unit_test
	valgrind ./cuda_unit_test

scaling_test: cuda_test
	./cuda_test "#s"

unit-test: qsopt_bin_test
	./qsopt_bin_test

all: cuda_test cuda_raytrace$(PT_BIN) cuda_raytrace_java.so

install: cuda_raytrace_java.so
	cp cuda_raytrace_java.so /usr/lib/cuda_raytrace_java.so

clean:
	rm -f $(BUILD)/*.o
	rm -f cuda_raytrace$(PT_BIN)
	rm -f cuda_raytrace_java.so
	rm -f cuda_test
