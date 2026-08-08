# BIG_N ?= 65536
BIG_N ?= 32768
# BIG_N ?= 16384
# BIG_N ?= 8192
# BIG_N ?= 2048
# BIG_N ?= 1024
# BIG_N ?= 512
# BIG_N ?= 4

REP_SIZE ?= 512

BLAS_LIB ?= blis
TOOLSET ?= gnu

ifeq ($(TOOLSET), intel)
	FFLAGS = ifx -O3 -qopenmp -qopt-report=3 -xHost
else ifeq ($(TOOLSET), gnu)
	FFLAGS = gfortran -O3 -fopenmp -march=native -mtune=native -ffast-math -ftree-vectorize
endif

ifeq ($(BLAS_LIB), mkl)
	ifeq ($(TOOLSET), intel)
		BLAS_FLAGS = -qmkl=parallel
	else ifeq ($(TOOLSET), gnu)
		BLAS_FLAGS = -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm 
	endif
else ifeq ($(BLAS_LIB), blis)
	BLAS_FLAGS = -L/home/andersx/opt/blis/lib -lblis
	# BLAS_FLAGS +=  -L/home/andersx/opt/libflame/lib -lflame
	LD_FLAGS = LD_LIBRARY_PATH=/home/andersx/opt/blis/lib:/home/andersx/opt/libflame/lib:${LD_LIBRARY_PATH}
endif

N_FEATURES ?= 4096

# CUDA GPU target
# Override CUDA_ARCH for your GPU, e.g.: make rff_gpu CUDA_ARCH=sm_86
CUDA_ARCH ?= sm_80
N_ATOMS ?= 5

all: test_symm test_asymm

kernel_symm: kernel.f90
	${FFLAGS} kernel.f90 ${BLAS_FLAGS} -o kernel_symm

test_symm: kernel_symm
	OMP_NUM_THREADS=8 BLIS_NUM_THREADS=8 MKL_NUM_THREADS=8 ${LD_FLAGS} ./kernel_symm ${BIG_N} ${REP_SIZE}

kernel_asymm: kernel_asymm.f90
	${FFLAGS} kernel_asymm.f90 ${BLAS_FLAGS} -o kernel_asymm

test_asymm: kernel_asymm
	OMP_NUM_THREADS=8 BLIS_NUM_THREADS=8 MKL_NUM_THREADS=8 ${LD_FLAGS} ./kernel_asymm ${BIG_N} ${REP_SIZE}

rff: rff.f90
	${FFLAGS} rff.f90 ${BLAS_FLAGS} -o rff

test_rff: rff
	OMP_NUM_THREADS=8 BLIS_NUM_THREADS=8 MKL_NUM_THREADS=8 ${LD_FLAGS} ./rff ${BIG_N} ${REP_SIZE} ${N_FEATURES}

rff_forces: rff_forces.f90
	${FFLAGS} rff_forces.f90 ${BLAS_FLAGS} -o rff_forces

test_rff_forces: rff_forces
	OMP_NUM_THREADS=8 BLIS_NUM_THREADS=8 MKL_NUM_THREADS=8 ${LD_FLAGS} ./rff_forces ${BIG_N} ${REP_SIZE} ${N_FEATURES}

sorf: sorf.f90
	${FFLAGS} sorf.f90 ${BLAS_FLAGS} -o sorf

test_sorf: sorf
	OMP_NUM_THREADS=8 BLIS_NUM_THREADS=8 MKL_NUM_THREADS=8 ${LD_FLAGS} ./sorf ${BIG_N} ${REP_SIZE} ${N_FEATURES}

rff_gpu: rff_gpu.cu
	nvcc -O3 -arch=$(CUDA_ARCH) rff_gpu.cu -lcublas -lcusolver -o rff_gpu

test_rff_gpu: rff_gpu
	./rff_gpu ${BIG_N} ${REP_SIZE} ${N_FEATURES} ${N_ATOMS}

KRR_N    ?= 500
KRR_TEST ?= 100
KRR_M    ?= 36
KRR_D    ?= 15

krr_gpu: krr_gpu.cu
	nvcc -O3 -arch=$(CUDA_ARCH) krr_gpu.cu -lcublas -lcusolver -o krr_gpu

test_krr_gpu: krr_gpu
	./krr_gpu ${KRR_N} ${KRR_TEST} ${KRR_M} ${KRR_D}

.PHONY: clean
clean:
	rm -rf test_kernel kernel_symm kernel_asymm rff rff_forces sorf rff_gpu krr_gpu

# gfortran_amd_symm:
# 	gfortran -O3 -fopenmp -march=native -mtune=native -ffast-math -ftree-vectorize kernel.f90 -L/home/andersx/opt/blis/lib  -L/home/andersx/opt/libflame/lib -lblis -o test_kernel 
# 
# test_amd:
# 	LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.2/lib:/home/andersx/opt/blis/lib:/home/andersx/opt/libflame/lib:${LD_LIBRARY_PATH} OMP_NUM_THREADS=8 BLIS_NUM_THREADS=8 ./test_kernel ${BIG_N} ${REP_SIZE} 
#
