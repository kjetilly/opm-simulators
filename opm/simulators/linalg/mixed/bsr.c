#define _POSIX_C_SOURCE 200112L
#include "bsr.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#pragma GCC push_options

bsr_matrix* bsr_alloc()
{
    bsr_matrix *A=malloc(sizeof(bsr_matrix));
    A->nrows = 0;
    A->ncols = 0;
    A->nnz   = 0;
    A->b     = 0;

    A->rowptr = NULL;
    A->colidx = NULL;
    A->dbl    = NULL;
    A->flt    = NULL;

    return A;
}

void bsr_free(bsr_matrix *A)
{
    if(A==NULL) return;

    if(A->flt    != NULL) free(A->flt);
    if(A->dbl    != NULL) free(A->dbl);
    if(A->colidx != NULL) free(A->colidx);
    if(A->rowptr != NULL) free(A->rowptr);

    free(A);
    A=NULL;
}

void bsr_init(bsr_matrix *A, int nrows, int nnz, int b)
{
    A->nrows=nrows;
    A->ncols=nrows;
    A->nnz=nnz;
    A->b=b;

    A->rowptr = malloc((nrows+1)*sizeof(int));
    A->colidx = malloc(nnz*sizeof(int));
    A->dbl    = malloc(b*b*nnz*sizeof(double));
    A->flt    = malloc(b*b*nnz*sizeof(float));
}

void bsr_info(bsr_matrix *A)
{
    printf("nrows = %d\n",A->nrows);
    printf("ncols = %d\n",A->ncols);
    printf("nnz   = %d\n",A->nnz);
    printf("b     = %d\n",A->b);

    printf("rowptr= 0x%08lX\n",(uint64_t)A->rowptr);
    printf("colidx= 0x%08lX\n",(uint64_t)A->colidx);
    printf("dbl   = 0x%08lX\n",(uint64_t)A->dbl);

    printf("\n");
}

void bsr_vmspmv3(bsr_matrix *A, const double *x, double *y)
{
    int nrows = A->nrows;
    int *rowptr=A->rowptr;
    int *colidx=A->colidx;
    const float *data=A->flt;

    const int b=3;

    for(int i=0;i<nrows;i++)
    {
        double result[3] = {0.0, 0.0, 0.0};
        
        for(int k=rowptr[i];k<rowptr[i+1];k++)
        {
            const float *AA=data+9*k;
            int j = colidx[k];
            const double *xj = x+b*j;
            
            // Matrix-vector multiply: result += AA * xj (with float to double conversion)
            for(int row=0;row<3;row++)
            {
                result[row] += (double)AA[row] * xj[0] + (double)AA[3+row] * xj[1] + (double)AA[6+row] * xj[2];
            }
        }

        // Store result
        double *y_i = y+b*i;
        for(int m=0;m<3;m++) y_i[m] = result[m];
    }
}

void bsr_vdspmv3(bsr_matrix *A, const double *x, double *y)
{
    int nrows = A->nrows;
    int *rowptr=A->rowptr;
    int *colidx=A->colidx;
    const double *data=A->dbl;

    const int b=3;

    for(int i=0;i<nrows;i++)
    {
        double result[3] = {0.0, 0.0, 0.0};
        
        for(int k=rowptr[i];k<rowptr[i+1];k++)
        {
            const double *AA=data+9*k;
            int j = colidx[k];
            const double *xj = x+b*j;
            
            // Matrix-vector multiply: result += AA * xj
            for(int row=0;row<3;row++)
            {
                result[row] += AA[row] * xj[0] + AA[3+row] * xj[1] + AA[6+row] * xj[2];
            }
        }

        // Store result
        double *y_i = y+b*i;
        for(int m=0;m<3;m++) y_i[m] = result[m];
    }
}



void bsr_downcast(bsr_matrix *M)
{
    int nnz = M->nnz;
    int b = M->b;

    if(M->flt==NULL) posix_memalign((void**)&(M->flt),64,b*b*nnz*sizeof(float));
    for(int i=0;i<b*b*nnz;i++) M->flt[i]=M->dbl[i];
}


void bsr_sparsity(const bsr_matrix *A, const char *name)
{
    printf("%s =\n[\n",name);
    int count=1;
    int offset=0;
    for(int i=0; i<A->nrows; i++)
    {
        printf("%4d: ",offset);
        for(int j=A->rowptr[i];j<A->rowptr[i+1];j++)
        {
            printf(" %4d",A->colidx[j]);
            offset++;
        }
        printf("\n");
        count++;
        if(count>16) break;
    }
    printf("]\n");
}

void bsr_nonzeros(bsr_matrix *A, const char *name)
{
    printf("%s =\n[\n",name);
    int count=1;
    int b=A->b;
    int bb=b*b;
    for(int i=0; i<A->nrows; i++)
    {
        for(int j=A->rowptr[i];j<A->rowptr[i+1];j++)
        {
            printf("|");
            for(int m=0;m<b;m++)
            {
                for(int n=0;n<b;n++)
                {
                    printf(" %+.4e",A->dbl[j*bb + m*b + n]);
                }
                printf(" |");
            }
            printf("\n");
        }
        count++;
        if(count>6) break;
        printf("\n");
    }
    printf("]\n");
}

#pragma GCC pop_options

