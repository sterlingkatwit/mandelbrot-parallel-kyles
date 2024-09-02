#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>

extern void matToImage(char* filename, int* mat, int* dims);
extern void matToImageColor(char* filename, int* mat, int* dims);
int main(int argc, char **argv){

    int rank, numranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status stat;

    int nx=24000;
    int ny=16000;
    // int nx=1200;
    // int ny=800;

    int* matrix=(int*)malloc(nx*ny*sizeof(int));

    int maxIter=255;
    double xStart=-2;
    double xEnd=1;
    double yStart=-1;
    double yEnd=1;

    double x=0;
    double y=0;
    double x0=0;
    double y0=0;
    int iter=0;


    double startT,endT;
    double nodeTS,nodeTE,nodeTT;

    int i;
    int start, end, nextStart, 

        //**Intended to store the start/end point of the part of the matrix thats being worked on. Need a value specific to the part that the node is currently working on.
        //**Based off of the matrix[i*nx+j]
        //doesnt do anything of significance anymore.
        currentS;

    // splitting work up using ny, so each node will do this number of ny at a time. Seems like a simple way to split the work.
    int numNY = 120;
    bool done = false;
    int doneRanks = 0;

    // holds values before putting them in main matrix.
    int* tempMat=(int*)malloc(nx*ny*sizeof(int));

    // master region
    if(rank==0){
        nextStart=0;
        startT = MPI_Wtime();
        for(int i=1;i<numranks;i++){
            start=nextStart;
            end=start+numNY-1;
            nextStart+=numNY;
            
            MPI_Send(&start,1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(&end,1,MPI_INT,i,0,MPI_COMM_WORLD);
        }
        
        while(!done){
            // receives to keep looping, actual content doesnt do anything
            MPI_Recv(&currentS,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&stat);
            //printf("got currentS: %d", currentS);
	    
            if(nextStart>ny){ 
                start=-1;
                doneRanks++;             
            }
            else {
                start=nextStart;
                end=start+numNY-1;
                nextStart+=numNY;
            }
            if(end>ny){
                end=ny;
            }

            MPI_Send(&start,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD);
            MPI_Send(&end,1,MPI_INT,stat.MPI_SOURCE,0,MPI_COMM_WORLD);
            if(doneRanks==numranks-1){
                done=true;
            }
        }
    }

    // workers region
    if(rank!=0){
        while(true){
            MPI_Recv(&start,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat); 
            MPI_Recv(&end,1,MPI_INT,0,0,MPI_COMM_WORLD,&stat); 
            if(start==-1){
                printf("Rank: %d done, Total time = %.3f\n", rank, nodeTT);
                break;
            }        
            nodeTS = MPI_Wtime();
            //printf("Rank: %d start. start= %d, end = %d\n", rank, start, end);
            
            // starts thread region
            #pragma omp parallel private(x0, y0, x, y, iter) num_threads(12)
            {
                double thrTS,thrTE;
                double* thrTT = malloc(omp_get_num_threads()*sizeof(double));
                // this doesnt do anything directly anymore with mpi_gather version.
                // used in a send to master region otherwise loop doesnt work.
                // also starting thread region right before loop messes up result, so this prevents that.
                currentS = 0;
                // mandelbrot set algorithm, open_mp for prob goes here
                #pragma omp for schedule(static, 10) nowait
                for(int h=start;h<end+1;h++){
                    thrTS = omp_get_wtime();
                    // splits into threads
                    for(int j=0;j<nx;j++){
                        x0=xStart+(1.0*j/nx)*(xEnd-xStart);
                        y0=yStart+(1.0*h/ny)*(yEnd-yStart);

                        x=0;
                        y=0;
                        iter=0;
                        while(iter<maxIter){
                            iter++;

                            double temp=x*x-y*y+x0;
                            y=2*x*y+y0;
                            x=temp;
                            
                            if(x*x+y*y>4){
                                break;
                            }
                        }
                        tempMat[h*nx+j]=iter;
                    }
                }
                if(start==0){
                    thrTE = omp_get_wtime();
                    thrTT[omp_get_thread_num()] = (thrTE-thrTS);
                    printf("Thread %d time: %.10f\n", omp_get_thread_num(), thrTT[omp_get_thread_num()]);
                }
            }
            nodeTE = MPI_Wtime();
            nodeTT += (nodeTE-nodeTS);
            // originially sends back region completed in this set and the completed portion of matrix
            // now just sends a value so the master keeps looping, the value is irrelevant
            MPI_Send(&currentS,1,MPI_INT,0,0,MPI_COMM_WORLD);
        }  
    }
    endT = MPI_Wtime();
    // brings all ranks' tempMats together in matrix
    MPI_Reduce(tempMat, matrix, nx*ny, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    int dims[2];
    dims[0]=ny;
    dims[1]=nx;

    // creates the image
    if(rank==0){
        matToImage("mandelbrot.jpg", matrix, dims);
        printf("Calc Time = %.3f\n", (endT-startT));
    }

    MPI_Finalize();
    return 0;
}



