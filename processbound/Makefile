process.o: process.cu
	nvcc -g -G process.cu -o process -arch=compute_20

ptx: process.cu
	nvcc --ptx -arch sm_20 process.cu

#OTHER
clean:
	rm -f *~ *.o *.swp process
