#include "prec.h"

#include <stdio.h>
#include <stdlib.h>

#pragma GCC push_options


prec_t *prec_alloc()
{
    prec_t *P = malloc(sizeof(prec_t));
    P->L=bsr_alloc();
    P->D=bsr_alloc();
    P->U=bsr_alloc();

    P->noffsets=-1;
    P->offsets=NULL;

    return P;
}

void prec_free(prec_t *P)
{
    if(P==NULL) return;

    if(P->offsets!=NULL) free(P->offsets);
    bsr_free(P->U);
    bsr_free(P->D);
    bsr_free(P->L);

    free(P);
    P=NULL;
}

int prec_analyze(bsr_matrix *M, int (*offsets)[3])
{
    int count=0;
    for(int i=0;i<M->nrows;i++)
    {
        for(int z=M->rowptr[i];z<M->rowptr[i+1];z++)
        {
            int j = M->colidx[z];
            int match=0;
            for(int m=M->rowptr[j];m<M->rowptr[j+1];m++)
            {
                int k = M->colidx[m];
                for(int n=M->rowptr[i];n<M->rowptr[i+1];n++)
                {
                    int jjj=M->colidx[n];
                    match += (k==jjj);
                    if(k==jjj)
                    {
                        if(offsets)
                        {
                            offsets[count][0]=z;//ij
                            offsets[count][1]=n;//ik
                            offsets[count][2]=m;//jk
                        }
                        count++;
                    }
                }
            }
        }
    }
    return count;
}


void prec_init(prec_t *P, bsr_matrix const *A)
{
    int b     = A->b;
    int nrows = A->nrows;
    int nnz   = A->nnz;

    bsr_matrix *L=P->L;
    bsr_matrix *D=P->D;
    bsr_matrix *U=P->U;

    bsr_init(P->L, nrows, (nnz-nrows)/2, b);
    bsr_init(P->D, nrows, nrows, b);
    bsr_init(P->U, nrows, (nnz-nrows)/2, b);

    // Sparsity of L,D, and U components of matrix A.
    // Sparsity of L is expressed for its transpose
    // which is identical to the sparsity of U due to
    // the assumption of structural symmetry.
    nnz=0;
    for (int i=0;i<nrows;i++)
    {
        L->rowptr[i]=nnz;
        D->rowptr[i]=i;
        U->rowptr[i]=nnz;
        for (int k=A->rowptr[i];k<A->rowptr[i+1];k++)
        {
            int j=A->colidx[k];
            if(j>i)
            {
                L->colidx[nnz]=j;
                U->colidx[nnz]=j;
                nnz++;
            }
        }
        D->colidx[i]=i;
    }
    L->rowptr[nrows]=nnz;
    D->rowptr[nrows]=nrows;
    U->rowptr[nrows]=nnz;

    // replace with asserts?
    L->nnz=nnz;
    D->nnz=nrows;
    U->nnz=nnz;


    // offsets for off-diagonal updates
    int count;
    count = prec_analyze(U,NULL);
    P->offsets = malloc(3*(count+1)*sizeof(int));
    count = prec_analyze(U,P->offsets);
    P->offsets[count][0]=U->nnz;
    P->noffsets=count;
}


/*
void mat3_view(double const *M, char const *name)
{
    printf("%s = \n[\n",name);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            printf("%+.4e ",M[3*j+i]);
        }
        printf("\n");
    }
    printf("\n]\n");
}
*/
/*
void mat3_T(double *B, double const *A)
{
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) B[3*i+j] = A[i+3*j];
}
*/
void mat3_matmul(double *C, const double *A, const double *B)
{
    // assume 3x3 column-major matrices
    // account for possiblity of C and A referring to same memory location
    double M[9];
    for(int k=0;k<9;k++) M[k]=0.0;
    for(int j=0;j<3;j++)
    {
        double *m_j = M+3*j;        // j-th column of M
        for(int k=0;k<3;k++)
        {
            double b_kj = B[k+3*j]; // kj-th element of B
            double const *a_k = A+3*k;    // k-th column of A
            for(int i=0;i<3;i++) m_j[i] += a_k[i]*b_kj; // |m_j> += |a_k> * b_kj;
        }
    }
    for(int k=0;k<9;k++) C[k]=M[k];
}

void mat3_matfms(double *C, const double *A, const double *B)
{
    // assume 3x3 column-major matrices
    // account for possiblity of C and A referring to same memory location
    double M[9];
    for(int k=0;k<9;k++) M[k]=0.0;
    for(int j=0;j<3;j++)
    {
        double *m_j = M+3*j;        // j-th column of M
        for(int k=0;k<3;k++)
        {
            double b_kj = B[k+3*j]; // kj-th element of B
            double const *a_k = A+3*k;    // k-th column of A
            for(int i=0;i<3;i++) m_j[i] += a_k[i]*b_kj; // |m_j> += |a_k> * b_kj;
        }
    }
    for(int k=0;k<9;k++) C[k]-=M[k];
}


void mat3_inv(double *invA, const double *A)
{
    double M[9];
    M[0] = +(A[4]*A[8]-A[5]*A[7]);
    M[3] = -(A[3]*A[8]-A[5]*A[6]);
    M[6] = +(A[3]*A[7]-A[4]*A[6]);
    M[1] = -(A[1]*A[8]-A[2]*A[7]);
    M[4] = +(A[0]*A[8]-A[2]*A[6]);
    M[7] = -(A[0]*A[7]-A[1]*A[6]);
    M[2] = +(A[1]*A[5]-A[2]*A[4]);
    M[5] = -(A[0]*A[5]-A[2]*A[3]);
    M[8] = +(A[0]*A[4]-A[1]*A[3]);

    double detA = A[0]*M[0]+A[1]*M[3]+A[2]*M[6];
    for(int k=0;k<9;k++) invA[k]=M[k]/detA;
}

inline void vec_copy9(double *y, double const *x)
{
    for(int i=0;i<9;i++) y[i]=x[i];
}

void mat3_rmul(double *A, double const *B)
{
    // Compute A = A * B (right multiplication)
    // Both matrices are 3x3 in column-major format
    double temp[9];
    
    for(int j=0;j<3;j++)
    {
        // Compute column j of result
        for(int i=0;i<3;i++)
        {
            temp[3*j+i] = A[i]*B[3*j] + A[3+i]*B[3*j+1] + A[6+i]*B[3*j+2];
        }
    }
    
    // Copy result back to A
    for(int k=0;k<9;k++) A[k] = temp[k];
}

void mat3_lmul(double const *A, double *B)
{
    // Compute B = A * B (left multiplication)
    // Both matrices are 3x3 in column-major format
    double temp[9];
    
    for(int j=0;j<3;j++)
    {
        // Compute column j of result
        for(int i=0;i<3;i++)
        {
            temp[3*j+i] = A[i]*B[3*j] + A[3+i]*B[3*j+1] + A[6+i]*B[3*j+2];
        }
    }
    
    // Copy result back to B
    for(int k=0;k<9;k++) B[k] = temp[k];
}

void mat3_vfms(double *C, double const *A, double const *B)
{
    // Compute C -= A * B (fused multiply-subtract)
    // A is 3x3 matrix, B is 3x3 matrix (column-major)
    for(int j=0;j<3;j++)
    {
        // Compute column j of A*B and subtract from C
        for(int i=0;i<3;i++)
        {
            C[3*j+i] -= A[i]*B[3*j] + A[3+i]*B[3*j+1] + A[6+i]*B[3*j+2];
        }
    }
}


void prec_dilu_factorize(prec_t *P, bsr_matrix *A)
{
    int nrows = A->nrows;
    int b     = A->b;
    int bb    = b*b;

    bsr_matrix *L=P->L;
    bsr_matrix *D=P->D;
    bsr_matrix *U=P->U;

    // Splitting values of A into L, D, and U, respectively
    int kU=0;
    for(int i=0;i<nrows;i++)
    {
        for (int k=A->rowptr[i];k<A->rowptr[i+1];k++)
        {
            int j=A->colidx[k];
            if(j<i)       // struct-transpose of L
            {
                int kL = L->rowptr[j];
                vec_copy9(L->dbl + bb*kL, A->dbl + bb*k);
                L->rowptr[j]++;
            }
            else if(j==i) // struct-copy of D
            {
                vec_copy9(D->dbl + bb*i, A->dbl + bb*k);
            }
            else if(j>i) // struct-copy of U
            {
                vec_copy9(U->dbl + bb*kU, A->dbl + bb*k);
                kU++;
            }
        }
    }
    // reset rowptr of L
    for(int i=nrows;i>0;i--) L->rowptr[i]=L->rowptr[i-1];
    L->rowptr[0]=0;

    // Factorizing
    double scale[9]; //hard-coded to 3x3 blocks for now
    for(int i=0;i<A->nrows;i++)
    {
        mat3_inv(scale,D->dbl+i*bb);
        vec_copy9(D->dbl+bb*i, scale); //store inverse instead to simplify application
        for(int k=L->rowptr[i];k<L->rowptr[i+1];k++)
        {
            //scale column i of L
            mat3_rmul(L->dbl+k*bb,scale);

            //update diagonal of U
            int j=L->colidx[k];
            mat3_vfms(D->dbl+j*bb,L->dbl+k*bb,U->dbl+k*bb);

            //scale row i of U
            mat3_lmul(scale,U->dbl+k*bb);

            //NOT IMPLEMENTED!
            for(int m=L->rowptr[j];m<L->rowptr[j+1];m++)
            {
                if(L->colidx[m]==j)
                {
                    printf("ILU OFF_DIAGONALS NOT IMPLEMENTED!\n");
                    printf("(%d,%d)",m,j);
                    getchar();
                }
            }
        }
    }
}

void prec_ilu0_factorize(prec_t *P, bsr_matrix *A)
{
    int nrows = A->nrows;
    int b     = A->b;
    int bb    = b*b;

    bsr_matrix *L=P->L;
    bsr_matrix *D=P->D;
    bsr_matrix *U=P->U;

    // Splitting values of A into L, D, and U, respectively
    int kU=0;
    for(int i=0;i<nrows;i++)
    {
        for (int k=A->rowptr[i];k<A->rowptr[i+1];k++)
        {
            int j=A->colidx[k];
            if(j<i)       // struct-transpose of L
            {
                int kL = L->rowptr[j];
                vec_copy9(L->dbl + bb*kL, A->dbl + bb*k);
                L->rowptr[j]++;
            }
            else if(j==i) // struct-copy of D
            {
                vec_copy9(D->dbl + bb*i, A->dbl + bb*k);
            }
            else if(j>i) // struct-copy of U
            {
                vec_copy9(U->dbl + bb*kU, A->dbl + bb*k);
                kU++;
            }
        }
    }
    // reset rowptr of L
    for(int i=nrows;i>0;i--) L->rowptr[i]=L->rowptr[i-1];
    L->rowptr[0]=0;

    // Factorizing
    int idx=0;
    int next = P->offsets[idx][0];
    double scale[9]; //hard-coded to 3x3 blocks for now
    for(int i=0;i<A->nrows;i++)
    {
        mat3_inv(scale,D->dbl+i*bb);
        vec_copy9(D->dbl+bb*i, scale); //store inverse instead to simplify application
        for(int k=L->rowptr[i];k<L->rowptr[i+1];k++)
        {
            //scale column i of L
            mat3_rmul(L->dbl+k*bb,scale);

            //update diagonal D
            int j=L->colidx[k];
            mat3_vfms(D->dbl+j*bb,L->dbl+k*bb,U->dbl+k*bb);
        }

        while(next<U->rowptr[i+1])
        {
            int ij = P->offsets[idx][0];
            int ik = P->offsets[idx][1];
            int jk = P->offsets[idx][2];

            //update off-diagonals L and U
            mat3_vfms(U->dbl+jk*bb,L->dbl+ij*bb,U->dbl+ik*bb);
            mat3_vfms(L->dbl+jk*bb,L->dbl+ik*bb,U->dbl+ij*bb);

            //update marker
            next=P->offsets[++idx][0];
        }


        for(int k=L->rowptr[i];k<L->rowptr[i+1];k++)
        {
            //scale row i of U
            mat3_lmul(scale,U->dbl+k*bb);
        }

    }
}


inline void mat3_vecmul(const double *A, double *x)
{
    const int b=3;
    double z[3];
    for(int k=0;k<3;k++) z[k]=0;
    for(int c=0;c<b;c++)
    {
        for(int r=0;r<b;r++)
        {
            z[r]+=A[c*b+r]*x[c];
        }
    }
    for(int k=0;k<3;k++) x[k]=z[k];
}


inline void mat3_vecfms(double *y, const double *A, const double *x)
{
    const int b=3;
    double z[3];
    for(int k=0;k<3;k++) z[k]=0;
    for(int c=0;c<b;c++)
    {
        for(int r=0;r<b;r++)
        {
            z[r]+=A[b*c+r]*x[c];
        }
    }
    for(int k=0;k<3;k++) y[k]-=z[k];
}

void prec_apply3(prec_t *restrict P, double *x)
{
    //bsr_matrix *LT = &(P->LT);
    bsr_matrix const *L  = P->L;
    bsr_matrix const *D  = P->D;
    bsr_matrix const *U  = P->U;

    int b=L->b;
    int bb=b*b;

    // Lower triangular solve assuming ones on diagonal
    for(int i=0;i<L->ncols;i++)
    {
        for(int k=L->rowptr[i];k<L->rowptr[i+1];k++)
        {
            int j=L->colidx[k];
            mat3_vecfms(x+b*j,L->dbl+k*bb,x+b*i);
        }
    }

    // Muliply by (inverse) diagonal matrix
    for (int i=0;i<D->ncols;i++) mat3_vecmul(D->dbl+bb*i,x+b*i);

    // Upper triangular solve assuming ones on diagonal
    for(int i=U->ncols;i>0;i--)
    {
        for(int k=U->rowptr[i]-1;k>U->rowptr[i-1]-1;k--)
        {
            int j=U->colidx[k];
            mat3_vecfms(x+b*(i-1),U->dbl+k*bb,x+b*j);
        }
    }
}



void prec_mapply3c(prec_t *restrict P, double *x)
{
    bsr_matrix *L  = P->L;
    bsr_matrix *D  = P->D;
    bsr_matrix *U  = P->U;

    int b=L->b;
    int bb=b*b;

    // Lower triangular solve assuming ones on diagonal
    for(int i=0;i<L->ncols;i++)
    {
        double xi[3];
        for(int m=0;m<3;m++) xi[m] = x[b*i+m];
        
        for(int k=L->rowptr[i];k<L->rowptr[i+1];k++)
        {
            const float *A = L->flt+k*bb;
            int j=U->colidx[k]; // should be L, but does not matter due to structural symmetry
            
            // Matrix-vector multiply: result = A * xi (converted to double)
            double result[3];
            for(int row=0;row<3;row++)
            {
                result[row] = (double)A[row] * xi[0] + (double)A[3+row] * xi[1] + (double)A[6+row] * xi[2];
            }
            
            // xj -= result
            double *xj = x+b*j;
            for(int n=0;n<3;n++) xj[n] -= result[n];
        }

        // Multiply by (inverse) diagonal block
        const float *A = D->flt+i*bb;
        double result[3];
        for(int row=0;row<3;row++)
        {
            result[row] = (double)A[row] * xi[0] + (double)A[3+row] * xi[1] + (double)A[6+row] * xi[2];
        }
        for(int m=0;m<3;m++) x[b*i+m] = result[m];
    }

    // Upper triangular solve assuming nonzeros stored in original order
    for(int i=U->ncols;i>0;i--)
    {
        double result[3] = {0.0, 0.0, 0.0};
        
        for(int k=U->rowptr[i]-1;k>U->rowptr[i-1]-1;k--)
        {
            const float *A = U->flt+k*bb;
            int j=U->colidx[k];
            double *xj = x+b*j;
            
            // Accumulate A * xj
            for(int row=0;row<3;row++)
            {
                result[row] += (double)A[row] * xj[0] + (double)A[3+row] * xj[1] + (double)A[6+row] * xj[2];
            }
        }
        
        double *xi = x+b*(i-1);
        for(int m=0;m<3;m++) xi[m] -= result[m];
    }
}

void prec_dapply3c(prec_t *restrict P, double *x)
{
    bsr_matrix *L  = P->L;
    bsr_matrix *D  = P->D;
    bsr_matrix *U  = P->U;

    int b=L->b;
    int bb=b*b;

    // Lower triangular solve assuming ones on diagonal
    for(int i=0;i<L->ncols;i++)
    {
        double xi[3];
        for(int m=0;m<3;m++) xi[m] = x[b*i+m];
        
        for(int k=L->rowptr[i];k<L->rowptr[i+1];k++)
        {
            const double *A = L->dbl+k*bb;
            int j=U->colidx[k];
            
            // Matrix-vector multiply: result = A * xi
            double result[3];
            for(int row=0;row<3;row++)
            {
                result[row] = A[row] * xi[0] + A[3+row] * xi[1] + A[6+row] * xi[2];
            }
            
            // xj -= result
            double *xj = x+b*j;
            for(int kk=0;kk<3;kk++) xj[kk] -= result[kk];
        }

        // Multiply by (inverse) diagonal block
        const double *A = D->dbl+i*bb;
        double result[3];
        for(int row=0;row<3;row++)
        {
            result[row] = A[row] * xi[0] + A[3+row] * xi[1] + A[6+row] * xi[2];
        }
        for(int m=0;m<3;m++) x[b*i+m] = result[m];
    }

    // Upper triangular solve assuming nonzeros stored in original order
    for(int i=U->ncols;i>0;i--)
    {
        double result[3] = {0.0, 0.0, 0.0};
        
        for(int k=U->rowptr[i]-1;k>U->rowptr[i-1]-1;k--)
        {
            const double *A = U->dbl+k*bb;
            int j=U->colidx[k];
            double *xj = x+b*j;
            
            // Accumulate A * xj
            for(int row=0;row<3;row++)
            {
                result[row] += A[row] * xj[0] + A[3+row] * xj[1] + A[6+row] * xj[2];
            }
        }
        
        double *xi = x+b*(i-1);
        for(int m=0;m<3;m++) xi[m] -= result[m];
    }
}








void prec_downcast(prec_t *P)
{
    bsr_downcast(P->L);
    bsr_downcast(P->D);
    bsr_downcast(P->U);
}



void prec_info(prec_t *P)
{
    bsr_info(P->L);
    bsr_info(P->D);
    bsr_info(P->U);
}
#pragma GCC pop_options
