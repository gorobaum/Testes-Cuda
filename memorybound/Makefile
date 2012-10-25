memory.o: memory.cu
	nvcc memory.cu -o memory -arch=compute_20

ptx: memory.cu
	nvcc --ptx -arch sm_20 memory.cu

#OTHER
clean:
	rm -f *~ *.o *.swp memory
