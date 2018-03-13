#include <silo.h>
//#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <complex>
#include <math> 


template <typename T, int NX, int NY, int NZ> class GPESilo
{
public:
    // variable info
    int dims[]       = {NX, NY, NZ};
    int ndims        = 3;
    int nvars        = 3;
    int vardims[]    = {1, 1, 3};
    int centering[]  = {1, 1, 1};
    
    char *varnames[] = {"density", "phase", "probability_current"};
    T* density_tmp = NULL;
    T* phase_tmp   = NULL;
    T* jx_tmp      = NULL;
    T* jy_tmp      = NULL;
    T* jz_tmp      = NULL;
    
    // accuracy
    int datatype = 0;
    
    // mesh
    T  x[NX], y[NY], z[NZ];
    T* coords[] = {x, y, z};
    
    // .silo database handle
    DBfile *dbfile = NULL;
    
    
    // constructor
    GPESilo(std::string filename)
    {
        /* Open the Silo file */
        filename = filename + ".silo";
        dbfile = DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL,
                          "3D data representing wavefunction, density, phase, probability current vector",
                          DB_HDF5);
        if(dbfile == NULL)
        {
            fprintf(stderr, "Could not create Silo file!\n");
            exit(EXIt_FAILURE);
        }
        
        if      ( sizeof(T) == 4 ) datatype = DB_FLOAT;
        else if ( sizeof(T) == 8 ) datatype = DB_DOUBLE;
        else    { fprintf(stderr, "ERROR! Wrong datatype given in template parameter!\n"); exit(EXIT_FAILURE); }
        
        density_tmp = (T*) malloc(sizeof(T) * NX*NY*NZ );
        phase_tmp   = (T*) malloc(sizeof(T) * NX*NY*NZ );
        jx_tmp      = (T*) malloc(sizeof(T) * NX*NY*NZ );
        jy_tmp      = (T*) malloc(sizeof(T) * NX*NY*NZ );
        jz_tmp      = (T*) malloc(sizeof(T) * NX*NY*NZ );
        
        
        /* Define the mesh in Silo file */
        #pragma omp parallel sections
        {
            #pragma omp section
            for (int ix=0; ix < NX; ix++)  x[ix] = (T)(ix - NX/2);
            #pragma omp section
            for (int iy=0; iy < NY; iy++)  y[iy] = (T)(iy - NY/2);
            #pragma omp section
            for (int iz=0; iz < NZ; iz++)  z[iz] = (T)(iz - NZ/2);
        }
        DBPutQuadmesh(dbfile,"quadmesh",NULL,coords,dims,ndims,datatype,DB_COLLINEAR,NULL);
    }
    
    // destructor
    ~GPESilo()
    {
        // close .silo database
        DBClose(dbfile);
        free(density_tmp);
        free(phase_tmp);
        free(jx_tmp);
        free(jy_tmp);
        free(jz_tmp);
    }
    
    void write_psi(std:complex<T>* psi)
    {
        #pragma omp parallel for
        for (int ixyz=0; ixyz < NX*NY*NZ; ix++)
        {
            density_tmp[ixyz] = std::abs(psi[ixyz]);
            phase_tmp[ixyz]   = std::arg(psi[ixyz]);
        }
        
        
    }
};



