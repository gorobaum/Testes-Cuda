cuda: main.o matrix_kernel.o matrix_cuda.o 
	nvcc main.o matrix_kernel.o matrix_cuda.o -o matrix -arch compute_20
	
main.o: main.cpp matrix.h
	g++ main.cpp -Wall -ansi -g -c

matrix_cuda.o: matrix.cpp matrix_kernel.h matrix.h
	nvcc -c matrix.cpp -o matrix_cuda.o -arch compute_20

matrix_kernel.o: matrix_kernel.cu matrix_kernel.h
	nvcc -c matrix_kernel.cu -o matrix_kernel.o -arch compute_20


#OTHER
clean:
	rm -f *~ *.o rk
