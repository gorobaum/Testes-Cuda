cuda: main.o memory_kernel.o memory_cuda.o 
	nvcc -g -G main.o memory_kernel.o memory_cuda.o -o memory -arch compute_20
	
main.o: main.cpp memory.h
	nvcc -g -G main.cpp -g -c

memory_cuda.o: memory.cpp memory_kernel.h memory.h
	nvcc -g -G -c memory.cpp -o memory_cuda.o -arch compute_20

memory_kernel.o: memory_kernel.cu memory_kernel.h
	nvcc -g -G -c memory_kernel.cu -o memory_kernel.o -arch compute_20


#OTHER
clean:
	rm -f *~ *.o *.swp memory
