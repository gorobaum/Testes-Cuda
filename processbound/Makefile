process.o: process.cu
	nvcc process.cu -o process -arch=compute_20

ptx: process.cu
	nvcc --ptx -arch=compute_20 process.cu

#OTHER
clean:
	rm -f *~ *.o *.swp process
