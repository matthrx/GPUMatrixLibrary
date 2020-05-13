CUDA_ROOT_DIR=/usr/local/cuda
MAGMA_ROOT_DIR = /usr/local/magma
## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS= -Wall -O3  -ansi -pedantic -DHAVE_CUBLAS -DADD_ -std=c++11 -fopenmp -fpermissive
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= --expt-extended-lambda --verbose 
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
MAGMA_LIB_DIR = -L$(MAGMA_ROOT_DIR)/lib
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
MAGMA_INC_DIR = -I$(MAGMA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart -lcublas -lopenblas
MAGMA_LINK_LIBS = -lm -lmagma

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src
CUDA_DIR = src/cudaFiles
# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include
###############################################

## Make variables ##

# Target executable name:
EXE = library

# Object files:
OBJS = $(OBJ_DIR)/advancedOperationsInterface.o $(OBJ_DIR)/advancedOperationsInterfaceMagma.o $(OBJ_DIR)/generalInformation.o $(OBJ_DIR)/arithmeticOperationsInterface.o $(OBJ_DIR)/statisticOperationsInterface.o $(OBJ_DIR)/GpuMatrix.o
# VCU_FILES = vpath %.cu 

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
# install : 



$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) -o $@ $(OBJS) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(MAGMA_LIB_DIR) $(MAGMA_INC_DIR) $(MAGMA_LINK_LIBS)

# Compile main .cpp file to object files:
$(SRC_DIR)/.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@ 

# Compile C++ source files to object files:
$(OBJ_DIR)/GpuMatrix.o : $(SRC_DIR)/GpuMatrix.cpp $(SRC_DIR)/GpuMatrix.h
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/advancedOperationsInterfaceMagma.o : $(CUDA_DIR)/advancedOperations/advancedOperationsInterface.cpp
	$(CC) $(CC_FLAGS) $(MAGMA_INC_DIR) $(CUDA_INC_DIR) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/advancedOperationsInterface.o : $(CUDA_DIR)/advancedOperations/advancedOperationsInterface.cu $(CUDA_DIR)/advancedOperations/advancedOperationsKernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/statisticOperationsInterface.o : $(CUDA_DIR)/statisticOperations/statisticOperationsInterface.cu $(CUDA_DIR)/statisticOperations/statisticOperationsKernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/arithmeticOperationsInterface.o : $(CUDA_DIR)/arithmeticOperations/arithmeticOperationsInterface.cu $(CUDA_DIR)/arithmeticOperations/arithmeticOperationsKernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/generalInformation.o : $(CUDA_DIR)/generalInformation/generalInformation.cu 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)


# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
