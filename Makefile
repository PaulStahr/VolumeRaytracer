ifneq ($(wildcard /usr/local/cuda-8.0/.*),)
cxx_builtin_include_directory="/usr/local/cuda-8.0/include"
cuda_library_dir="/usr/local/cuda-8.0/nvvm/libdevice"
else
cxx_builtin_include_directory="/usr/lib/nvidia-cuda-toolkit/include"
cuda_library_dir="/usr/lib/nvidia-cuda-toolkit/libdevice"
endif

ifneq ($(wildcard /usr/lib/jvm/java-11-openjdk-amd64/include/.*),)
jni_library:=/usr/lib/jvm/java-11-openjdk-amd64/include/
else
jni_library:=/usr/lib/jvm/java-14-openjdk-amd64/include/
endif
#-I"${JAVA_HOME}/include" -I"${JAVA_HOME}/include/linux"
# -maxrregcount 36

PT_BIN := $(shell python3-config '--extension-suffix')
CFLAGS= -Wall -Wextra -pedantic -g -O2 -fopenmp -std=c++14 -lstdc++fs
LFLAGS= -lstdc++fs -lpng -ljpeg
CFLAGS += -DNDEBUG

SRC := src
BUILT := built
G++ := g++-7
CGCC := gcc-8

#debug: CCFLAGS += -DDEBUG -g

$(BUILT)/util.o: $(SRC)/util.cpp $(SRC)/util.h
	$(G++) $(SRC)/util.cpp -c $(CFLAGS) -o $(BUILT)/util.o -ljpeg -lpng -fPIC

$(BUILT)/image_io.o: $(SRC)/image_io.cpp $(SRC)/image_io.h
	$(G++) $(SRC)/image_io.cpp -c $(CFLAGS) -o $(BUILT)/image_io.o -ljpeg -lpng -fPIC

$(BUILT)/serialize.o: $(SRC)/serialize.cpp $(SRC)/serialize.h
	$(G++) $(SRC)/serialize.cpp -c $(CFLAGS) -o $(BUILT)/serialize.o -ljpeg -lpng -fPIC

$(BUILT)/raytracer.o: $(SRC)/cuda_volume_raytracer.cu $(SRC)/cuda_volume_raytracer.h
	nvcc -ccbin $(CGCC) -I$(cxx_builtin_include_directory) -D_FORCE_INLINES -O2 -v -c $(SRC)/cuda_volume_raytracer.cu -o $(BUILT)/raytracer.o --dont-use-profile -ldir=$(cuda_library_dir) --ptxas-options=-v -Xcompiler -fPIC -g -std=c++11 -Xcompiler -fopenmp -Xcompiler -msse -Xcompiler -msse2 -DNDEBUG

$(BUILT)/image_util.o: $(SRC)/image_util.cpp $(SRC)/image_util.h
	$(G++) $(SRC)/image_util.cpp -c $(CFLAGS) -o $(BUILT)/image_util.o -fPIC

$(BUILT)/io_util.o: $(SRC)/io_util.cpp $(SRC)/io_util.h
	$(G++) $(SRC)/io_util.cpp -c $(CFLAGS) -o $(BUILT)/io_util.o -fPIC

$(BUILT)/python_binding.o: $(SRC)/python_binding.cpp
	$(G++) $(SRC)/python_binding.cpp -c $(CFLAGS) -o $(BUILT)/python_binding.o -fPIC `python3 -m pybind11 --includes`

$(BUILT)/raytrace_test.o: $(SRC)/raytrace_test.cpp
	$(G++) $(SRC)/raytrace_test.cpp -c $(CFLAGS) -o $(BUILT)/raytrace_test.o -fPIC

$(BUILT)/java_binding.o: $(SRC)/java_binding.cpp $(SRC)/java_binding.h
	$(G++) $(SRC)/java_binding.cpp -c $(CFLAGS) -o $(BUILT)/java_binding.o -fPIC -I$(jni_library) -I$(jni_library)/linux

cuda_raytrace$(PT_BIN): $(BUILT)/image_io.o $(BUILT)/image_util.o $(BUILT)/raytracer.o $(BUILT)/serialize.o $(BUILT)/python_binding.o $(BUILT)/io_util.o $(BUILT)/serialize.o $(BUILT)/util.o
	$(G++) $(CFLAGS) $(BUILT)/image_io.o $(BUILT)/image_util.o $(BUILT)/raytracer.o $(BUILT)/serialize.o $(BUILT)/util.o $(BUILT)/io_util.o $(BUILT)/python_binding.o -lcudart  -L/usr/local/cuda-8.0/lib64/ -shared -fPIC -o cuda_raytrace$(PT_BIN) -I/usr/include/python3.6m/ ${LFLAGS}

cuda_raytrace_java.so: $(BUILT)/image_io.o $(BUILT)/image_util.o $(BUILT)/raytracer.o $(BUILT)/serialize.o $(BUILT)/util.o $(BUILT)/java_binding.o $(BUILT)/io_util.o $(BUILT)/serialize.o
	$(G++) $(CFLAGS) -lc $(BUILT)/image_io.o $(BUILT)/image_util.o $(BUILT)/raytracer.o $(BUILT)/util.o $(BUILT)/io_util.o $(BUILT)/serialize.o $(BUILT)/java_binding.o -lcudart  -L/usr/local/cuda-8.0/lib64/ -shared -fPIC -o cuda_raytrace_java.so -I$(jni_library) -I$(jni_library)/linux ${LFLAGS}

cuda_test: $(BUILT)/image_util.o $(BUILT)/image_io.o $(BUILT)/raytracer.o $(BUILT)/raytrace_test.o $(BUILT)/io_util.o $(BUILT)/serialize.o $(BUILT)/util.o
	$(G++) $(CFLAGS) $(BUILT)/image_util.o $(BUILT)/image_io.o $(BUILT)/raytracer.o $(BUILT)/serialize.o $(BUILT)/util.o $(BUILT)/io_util.o $(BUILT)/raytrace_test.o -lcudart  -L/usr/local/cuda-8.0/lib64/ -fPIC -o cuda_test ${LFLAGS}

scaling_test: cuda_test
	./cuda_test "#s"

all: cuda_test cuda_raytrace$(PT_BIN) cuda_raytrace_java.so

install: cuda_raytrace_java.so
	cp cuda_raytrace_java.so /usr/lib/cuda_raytrace_java.so

clean:
	rm -f $(BUILT)/util.o
	rm -f $(BUILT)/image_io.o
	rm -f $(BUILT)/serialize.o
	rm -f $(BUILT)/raytracer.o
	rm -f $(BUILT)/image_util.o
	rm -f $(BUILT)/io_util.o
	rm -f $(BUILT)/java_binding.o
	rm -f $(BUILT)/python_binding.o
	rm -f $(BUILT)/raytrace_test.o
	rm -f cuda_raytrace$(PT_BIN)
	rm -f cuda_raytrace_java.so
	rm -f cuda_test
