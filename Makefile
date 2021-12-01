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

PT_BIN:= $(shell python3-config '--extension-suffix')
CCFLAGS:=-Wall -Wextra -g -O2 -fopenmp -std=c++11 -mavx2 -fPIC -Werror
CFLAGS:= -Wall -Wextra -pedantic -g -O2 -fopenmp -std=c++17 -mavx2 -fPIC -Werror
LFLAGS:= -lstdc++fs -lpng -ljpeg -fopenmp
LCFLAGS:= -lcuda -lcudart
CFLAGS += -DNDEBUG
CLIBS:= -lcudart  -L/usr/local/cuda-8.0/lib64/
SRC := src
BUILD := build
G++ := g++-7
CGCC := gcc-8
CC := g++

#debug: CCFLAGS += -DDEBUG -g

CFILES:=util.cpp image_io.cpp serialize.cpp image_util.cpp io_util.cpp
OFILES:=$(foreach f,$(CFILES),$(BUILD)/$(subst .cpp,.o,$f))

define built_object
$(BUILD)/$(subst .cpp,.o,$f): $(SRC)/$f $(SRC)/$(subst .cpp,.h,$f)
	$(CC) -c $(CFLAGS) $(DFLAGS) $(LDFLAGS) $(SRC)/$f -o $(BUILD)/$(subst .cpp,.o,$f)
endef
$(foreach f,$(CFILES),$(eval $(call built_object)))
$(foreach f,raytrace_test.cpp ,$(eval $(call built_object)))

$(BUILD)/raytracer.o: $(SRC)/cuda_volume_raytracer.cu $(SRC)/cuda_volume_raytracer.h $(SRC)/tuple_math.h $(SRC)/types.h
	$(CC) -D_FORCE_INLINES -O2 -c -x c++ $(SRC)/cuda_volume_raytracer.cu -o $@ $(CFLAGS) -msse -msse2 -DNCUDA

$(BUILD)/raytracer_cuda.o: $(SRC)/cuda_volume_raytracer.cu $(SRC)/cuda_volume_raytracer.h $(SRC)/tuple_math.h $(SRC)/types.h
	nvcc -ccbin $(CGCC) -I$(cxx_builtin_include_directory) -D_FORCE_INLINES -O2 -v -c $(SRC)/cuda_volume_raytracer.cu -o $@ --dont-use-profile -ldir=$(cuda_library_dir) --ptxas-options=-v $(foreach f,$(CCFLAGS), -Xcompiler $f) -DNDEBUG

$(BUILD)/test_main.o: $(SRC)/test_main.cpp $(SRC)/test_main.h $(SRC)/serialize_test.h $(SRC)/image_util_test.h $(SRC)/cuda_volume_raytracer_test.h
	$(CC) -c $(CFLAGS) $(DFLAGS) $(LDFLAGS) $(SRC)/test_main.cpp -o $@

$(BUILD)/python_binding.o: $(SRC)/python_binding.cpp
	$(G++) $(SRC)/python_binding.cpp -c $(CFLAGS) -o $(BUILD)/python_binding.o `python3 -m pybind11 --includes`

$(BUILD)/java_binding.o: $(SRC)/java_binding.cpp $(SRC)/java_binding.h
	$(CC) $(SRC)/java_binding.cpp -c $(CFLAGS) -o $(BUILD)/java_binding.o -I"${JAVA_HOME}" -I"${JAVA_HOME}/linux"

raytracer$(PT_BIN): $(OFILES) $(BUILD)/raytracer.o $(BUILD)/python_binding.o
	$(G++) $(OFILES) $(BUILD)/raytracer.o $(BUILD)/python_binding.o -shared -o $@ -I/usr/include/python3.6m/ $(LFLAGS)

raytracer_cuda$(PT_BIN): $(OFILES) $(BUILD)/raytracer.o $(BUILD)/python_binding.o
	$(G++) $(OFILES) $(BUILD)/raytracer.o $(BUILD)/python_binding.o $(CLIBS) -shared -o $@ -I/usr/include/python3.6m/ $(LFLAGS)

raytracer_java.so: $(OFILES) $(BUILD)/raytracer.o $(BUILD)/java_binding.o
	$(CC) -lc $(OFILES) $(BUILD)/raytracer.o $(BUILD)/java_binding.o -shared -o $@ $(LFLAGS)

raytracer_java_cuda.so: $(OFILES) $(BUILD)/raytracer_cuda.o $(BUILD)/java_binding.o
	$(CC) -lc $(OFILES) $(BUILD)/raytracer_cuda.o $(BUILD)/java_binding.o $(CLIBS) -shared -o $@ $(LFLAGS)

raytracer_test: $(OFILES) $(BUILD)/raytracer.o $(BUILD)/raytrace_test.o
	$(CC) $^ -o $@ $(LFLAGS)

raytracer_test_cuda: $(OFILES) $(BUILD)/raytracer_cuda.o $(BUILD)/raytrace_test.o
	$(CC) $^ $(CLIBS) -o $@ $(LFLAGS) $(LCFLAGS)

raytracer_unit_test: $(OFILES) $(BUILD)/raytracer.o $(BUILD)/test_main.o
	$(CC) $^ -lboost_unit_test_framework -no-pie $(LFLAGS) -o $@

raytracer_unit_test_cuda: $(OFILES) $(BUILD)/raytracer_cuda.o $(BUILD)/test_main.o
	$(CC) $^ $(CLIBS) -lboost_unit_test_framework -no-pie $(LFLAGS) $(LCFLAGS) -o $@

test: raytracer_unit_test
	valgrind ./raytracer_unit_test

test_cuda: raytracer_unit_test_cuda
	valgrind ./raytracer_unit_test_cuda	

scaling_test: raytracer_test
	./raytracer_test "#s"

all: raytracer$(PT_BIN) raytracer_cuda$(PT_BIN) raytracer_unit_test raytracer_unit_test_cuda raytracer_test raytracer_test_cuda raytracer_java.so raytracer_java_cuda.so test test_cuda

install: raytracer_java.so raytracer_java_cuda.so
	cp raytracer_java.so /usr/lib/raytracer_java.so
	cp raytracer_java_cuda.so /usr/lib/raytracer_java_cuda.so

clean:
	rm -f $(BUILD)/*.o
	rm -f raytracer$(PT_BIN)
	rm -f raytracer_cuda$(PT_BIN)
	rm -f raytracer_unit_test
	rm -f raytracer_unit_test_cuda
	rm -f raytracer_test
	rm -f raytracer_test_cuda
	rm -f raytracer_java.so
	rm -f raytracer_java_cuda.so
