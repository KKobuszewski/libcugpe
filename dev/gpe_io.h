#ifndef __GPE_IO_H__
#define __GPE_IO_H__

#include <stdio.h>
#include <stdlib.h>

#include "gpe_user_defined.h"

#define MAX_REC_LEN 256		// size of buffer for reading file


// TODO: make structure struct gpe_info
// struct gpe_additional_info + make use of operator ">>" ?

// TODO: Make it more felxible
static inline void save_info(   char* dftc_filename,
                                const int nx,
                                const int ny,
                                const int nz,
                                const double dt,
                                const int nom,
                                const double scat_lenght,
                                const double aspect_ratio,
                                const double omega_x,
                                const double r0,
                                const double npart)
{
    
    // Create dftc.info file
    char info_filename[256];
    sprintf( info_filename,"%s.info",dftc_filename);
    FILE* info_file = fopen(info_filename,"w");
    fprintf(info_file,"%d   # nx\n",nx);
    fprintf(info_file,"%d   # ny\n",ny);
    fprintf(info_file,"%d   # nz\n",nz);
    fprintf(info_file,"%lf   # dt\n",dt);
    fprintf(info_file,"%d   # nom\n",nom);
    fprintf(info_file,"%lf   # a_scattering\n",scat_lenght);
    fprintf(info_file,"%lf   # aspect_ratio\n",aspect_ratio);
    fprintf(info_file,"%lf   # omega_x\n",omega_x);
    fprintf(info_file,"%lf   # r0\n",r0);
    fprintf(info_file,"%lf   # npart\n",npart);
    fclose(info_file);
}

static inline void gpe_save_info(   char* dftc_filename,
                                    const int nx,
                                    const int ny,
                                    const int nz,
                                    const double dt,
                                    const int nom,
                                    const double scat_lenght,
                                    const double aspect_ratio,
                                    const double omega_x,
                                    const double r0,
                                    const double npart,
                                    const double total_time = 0)
{
    
    // Create dftc.info file
    char info_filename[256];
    sprintf( info_filename,"%s.info",dftc_filename);
    FILE* info_file = fopen(info_filename,"w");
    
#if (GPE_FOR == PARTICLES)
    fprintf(info_file,"%s   # system type\n","bosons");
    #ifdef DIPOLAR
    fprintf(info_file,"%s   # mode\n","dipolar");
    #else
    fprintf(info_file,"%s   # mode\n","contact");
    #endif
    
#elif (GPE_FOR == DIMERS)
    fprintf(info_file,"%s   # system type\n","fermions");
    #ifdef UNITARY
    fprintf(info_file,"%s   # mode\n","unitary");
    #else
    fprintf(info_file,"%s   # mode\n","bec");
    #endif
#else
    fprintf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nError describing system of simulation!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif
    
    fprintf(info_file,"%d   # nx\n",nx);
    fprintf(info_file,"%d   # ny\n",ny);
    fprintf(info_file,"%d   # nz\n",nz);
    fprintf(info_file,"%.15e   # dt\n",dt);
    fprintf(info_file,"%d   # nom\n",nom);
    fprintf(info_file,"%.15e   # a_scattering\n",scat_lenght);
    fprintf(info_file,"%.15e   # aspect_ratio\n",aspect_ratio);
    fprintf(info_file,"%.15e   # omega_x\n",omega_x);
    fprintf(info_file,"%.15e   # r0\n",r0);
    fprintf(info_file,"%.15e   # npart\n",npart);
    fprintf(info_file,"%.15e   # total time\n",total_time);
    fclose(info_file);
}


static inline void get_info(    char* info_filename,
                                int* nx,
                                int* ny,
                                int* nz,
                                double* dt,
                                int* nom,
                                double* scat_lenght,
                                double* aspect_ratio,
                                double* omega_x,
                                double* r0,
                                double* npart
                           )
{
    FILE *fp;
    fp=fopen(info_filename, "r");
    if(fp==NULL) {printf("Error: Cannot open file %s!\n",info_filename); exit(EXIT_FAILURE);}
    
    char s[MAX_REC_LEN]; // buffer for reading file
    
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , nx); // NX
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , ny); // NY
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , nz); // NZ
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", dt); // DT
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , nom);// NOM
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", scat_lenght);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", aspect_ratio);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", omega_x);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", r0);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", npart);
}


static inline void gpe_get_info(   char* dftc_filename,
									int* nx,
									int* ny,
									int* nz,
									double* dt,
									int* nom,
									double* scat_lenght,
									double* aspect_ratio,
									double* omega_x,
									double* r0,
									double* npart,
									double* total_time = NULL
							  )
{
    
    // Open dftc.info file
    char info_filename[256];
    sprintf( info_filename,"%s.info",dftc_filename);
    FILE* fp = fopen(info_filename,"r");
    if(fp==NULL) {printf("Error: Cannot open file %s!\n",info_filename); exit(EXIT_FAILURE);}
    
    char s[MAX_REC_LEN]; // buffer for reading file
    
    if(fgets(s, MAX_REC_LEN, fp) != NULL); // system type <string>                 (ignoring)
    if(fgets(s, MAX_REC_LEN, fp) != NULL); // mode                                 (ignoring)
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , nx); // NX
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , ny); // NY
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , nz); // NZ
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", dt); // DT
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%d %*s" , nom);// NOM
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", scat_lenght);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", aspect_ratio);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", omega_x);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", r0);
    if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", npart);
    if (total_time) if(fgets(s, MAX_REC_LEN, fp) != NULL) sscanf(s, "%lf %*s", total_time);
    fclose(fp);
}

#endif
