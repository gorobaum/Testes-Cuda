cuda: main.o memory_kernel.o memory_cuda.o 
	nvcc main.o memory_kernel.o memory_cuda.o -o memory -arch compute_20
	
main.o: main.cpp memory.h
	g++ main.cpp -Wall -ansi -g -c

memory_cuda.o: memory.cpp memory_kernel.h memory.h
	nvcc -c memory.cpp -o memory_cuda.o -arch compute_20

memory_kernel.o: memory_kernel.cu memory_kernel.h
	nvcc -c memory_kernel.cu -o memory_kernel.o -arch compute_20


#OTHER
clean:
	rm -f *~ *.o *.swp memory
