#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <complex>
//#include <cmath>
#include <math.h> 

#include <silo.h>


template <typename T, int _NX, int _NY, int _NZ> class GPESilo
{
public:
    int dberr;
    
    // variable info
    int dims[3]; // {NX, _NY, _NZ};
    int ndims;   // 3
    int nvars;   // 3
    
    char *varnames[3];        // {"density", "phase", "probability_current"};
    char *current_varnames[3];// {"jx","jy","jz"};
    T* density_tmp;
    T* phase_tmp;
    T* current_tmp[3];
    
    // accuracy
    int DB_DATATYPE;
    
    // mesh
    T  x[_NX], y[_NY], z[_NZ];
    T* coords[3];
    
    // .silo database handle
    DBfile *dbfile;
    
    
    // constructor
    GPESilo(std::string filename) : dberr(0)
    {
        ndims = 3;
        nvars = 3;
        dims[0] = _NX; varnames[0] = "density";
        dims[1] = _NY; varnames[1] = "phase";
        dims[2] = _NZ; varnames[2] = "probability_current";
        current_varnames[0] = "jx";
        current_varnames[1] = "jy";
        current_varnames[2] = "jz";
        coords[0] = x;
        coords[1] = y;
        coords[2] = z;
        density_tmp = NULL; phase_tmp = NULL; dbfile = NULL;
        
        /* Open the Silo file */
        filename = filename + ".silo";
        dbfile = DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL,
                          "3D data representing wavefunction, density, phase, probability current vector",
                          DB_HDF5);
        if(dbfile == NULL)
        {
            fprintf(stderr, "Could not create Silo file!\n");
            exit(EXIT_FAILURE);
        }
        
        if      ( sizeof(T) == 4 ) DB_DATATYPE = DB_FLOAT;
        else if ( sizeof(T) == 8 ) DB_DATATYPE = DB_DOUBLE;
        else    { fprintf(stderr, "ERROR! Wrong datatype given in template parameter!\n"); exit(EXIT_FAILURE); }
        
        density_tmp    = (T*) malloc(sizeof(T) * _NX*_NY*_NZ );
        phase_tmp      = (T*) malloc(sizeof(T) * _NX*_NY*_NZ );
        //current_tmp[0] = (T*) malloc(sizeof(T) * _NX*_NY*_NZ );
        //current_tmp[1] = (T*) malloc(sizeof(T) * _NX*_NY*_NZ );
        //current_tmp[2] = (T*) malloc(sizeof(T) * _NX*_NY*_NZ );
        
        
        /* Define the mesh in Silo file */
        #pragma omp parallel sections
        {
            #pragma omp section
            for (int ix=0; ix < _NX; ix++)  x[ix] = (T)(ix - _NX/2);
            #pragma omp section
            for (int iy=0; iy < _NY; iy++)  y[iy] = (T)(iy - _NY/2);
            #pragma omp section
            for (int iz=0; iz < _NZ; iz++)  z[iz] = (T)(iz - _NZ/2);
        }
        dberr = DBPutQuadmesh(dbfile,"quadmesh",NULL,coords,dims,ndims,DB_DATATYPE,DB_COLLINEAR,NULL);
    }
    
    // destructor
    ~GPESilo()
    {
        // close .silo database
        DBClose(dbfile);
        
        // clean memory
        free(density_tmp);
        free(phase_tmp);
        //free(current_tmp[0]);
        //free(current_tmp[1]);
        //free(current_tmp[2]);
    }
    
    void write_psi(std::complex<T>* psi)
    {
        #pragma omp parallel for
        for (int ixyz=0; ixyz < _NX*_NY*_NZ; ixyz++)
        {
            density_tmp[ixyz] = std::abs(psi[ixyz]);
            phase_tmp[ixyz]   = std::arg(psi[ixyz]);
        }
        
        /*dberr = DBWriteSlice(dbfile, varnames[0], density_tmp, datatype, int const *offset,
                            int cost *length, int const *stride, int const *dims, int ndims);*/
        dberr = DBPutQuadvar1(dbfile,varnames[0],"quadmesh",density_tmp,dims,ndims,NULL,0,DB_DATATYPE,DB_NODECENT,NULL);
        dberr = DBPutQuadvar1(dbfile,varnames[1],"quadmesh",phase_tmp,  dims,ndims,NULL,0,DB_DATATYPE,DB_NODECENT,NULL);
    }
    
     void write_currents(T* jx, T* jy, T* jz)
    {
        current_tmp[0] = jx;
        current_tmp[1] = jy;
        current_tmp[2] = jz;
//         #pragma omp parallel for
//         for (int ixyz=0; ixyz < _NX*_NY*_NZ; ix++)
//         {
//             density_tmp[ixyz] = std::abs(psi[ixyz]);
//             phase_tmp[ixyz]   = std::arg(psi[ixyz]);
//         }
        
        /*dberr = DBWriteSlice(dbfile, varnames[0], density_tmp, datatype, int const *offset,
                            int cost *length, int const *stride, int const *dims, int ndims);*/
        dberr = DBPutQuadvar(dbfile,varnames[2],"quadmesh",3,current_varnames,current_tmp,dims,ndims,NULL,0,DB_DATATYPE,DB_NODECENT,NULL);
    }
};

//template <> class GPESilo<double> {};
//template <> class GPESilo<float> {};

