/*
 * mylib.h
 *
 *  Created on: Apr 13, 2016
 *      Author: kunkun220189
 */

#ifndef MYLIB_H_
#define MYLIB_H_

#define DEBUG 	   0
#define INF 	   1e50
#define BIG_INT    100
#define EPS 	   1e-8
#define MAX_EPOCH  10
#define MAX_LACZOS 50

typedef double **  R_MAT;
typedef double *   R_VEC;

typedef int    **  I_MAT;
typedef int    *   I_VEC;


/*--------------------------------------------------------------------------*/
double fabs(double a);
double sqr(double x);
double min(double x, double y);
double max(double x, double y);
/*--------------------------------------------------------------------------*/

/*create a matrix with 0*/
I_MAT createIM(int n, int m);
I_VEC createIV(int n);
R_MAT createRM(int n, int m);
R_VEC createRV(int n);
/* clean vector*/
void destroyRM(R_MAT A, int n);
void destroyIM(I_MAT A, int n);
/*ret = a - b */
void   subtractVV(int d, R_VEC a, R_VEC b, R_VEC ret);
/*ret = a'*b */
double multVV(int d, R_VEC a, R_VEC b);
/*ret = v*a */
void multVS(int d, R_VEC a, double alpha);
/*M = M + alpha*b*b' */
void multMSV(int d, R_MAT M, R_VEC b, double alpha);
/* M = M*alpha */
void multMS(int d, R_MAT M, double alpha);
/* a'*M*b */
double multTriplet(int d, R_MAT M, R_VEC a, R_VEC b);
/* get trace of matrix M */
double getTrace(int d, R_MAT M);
/* (a-b)'*M*(a-b) */
double distanceVV(int d, R_MAT M, R_VEC a, R_VEC b);

/*--------------------------------------------------------------------------*/
/* ret = a'*a */
double norm(R_VEC a, int d);
/* random shuffle a vector u*/
void randperm(int n, I_VEC u);
typedef struct{
    int ind;
    double v;
} entry;
int cmpfunc (const void * a, const void * b);
/*--------------------------------------------------------------------------*/

struct Data{
	R_VEC X;       /* training example n x d*/
	I_VEC Y;       /* class labels  */

	int d;         /* # of features */
	int n;         /* # of examples */
	int nClass;    /* #of classes   */

	I_MAT  NegNeighbors; /* start by 1 */
	I_VEC  Targets;      /* start by 0 */
	I_VEC  Impostors;    /* start by 0 */

	int k1;      /* # of positive neighbors */
	int k2;      /* # of negative neighbors */

    R_VEC  temp; /* temporal array */

    R_VEC sim;   /* similarity of each example*/
    R_VEC kernel;/* kernel matrix for input */
};
typedef struct Data* pData;

R_VEC getData(pData input, int index);
I_VEC getTargets(pData input, int index);
double getKernel(pData input, int i, int j);

/* initialize the input data*/
pData initialData(R_VEC X, I_VEC Y, int d, int n, int nClass,
        int k1, int *pT, int k2, int *pI);
double getCost(R_MAT M, pData input);
double trainingAccuracy(pData input, R_MAT M);
void   clean(pData input);
/* update the similarity:
 * if   M == NULL, then sim[i] = sim[i] * alpha
 * if   M != NULL, then sim[i] = sim[i] + alpha * (u'*x[i])^2
 */
void updateStructure(pData input, R_MAT M, R_VEC u, double alpha);
void fastUpdateStructure(pData input, R_MAT M, int u, int v, double alpha);
int  getNearestImpostor(R_MAT M, pData input, int index, double *ret);
int  getFarthestTarget(R_MAT M, pData input, int index, double* dist);
/*--------------------------------------------------------------------------*/
/**
 * Lanczos method for finding approximate eigenvalues of matrix A of size dxd
 * The algorithm returns the tridiagonal matrix T, where
 * +) alpha is the diagonal of T
 * +) beta is the sub and sup diagonal of T
 * Q is the matrix of normalized output vectors
 * v is a temporal vector for random initial
 * A is the input vector
 * k is the desired number of iteration for Lanczos method
 */
int lanczos(int d, double** A, int k, double *alpha, double* beta, double** Q, double *v, double *temp);
/**
 * Rerurn the smallest eigenvalue and its corresponding eigenvector in v of
 * the matrix A
 * input:
 *        k : the desired number of iteration for Lanczos method
 *        A : the input matrix
 *        V : the output vector of Lanczos method (auxiliar)
 *        z : the eigenvectors of A (auxiliar)
 *        alpha, beta: input of the Lanczos method
 *        v : the output corresponding eigenvector
 */
double getSmallestEigenvalue(int k, int d, double** A, double *v, double *alpha,
        double * beta, double**V, double**z, double * temp, double ** X);



double learnAlpha(int d, R_MAT pM, R_MAT M, R_VEC a, R_VEC b,
		R_VEC k, R_VEC u, R_VEC tmp);

void updatePseudoInverse(int d, R_MAT pM, R_MAT M, R_VEC x,
		double alpha, R_VEC k, R_VEC u, R_VEC tmp);

double learnAlpha1(int d, R_MAT pM, R_MAT M, R_VEC a, R_VEC b,
		R_VEC k, R_VEC u, R_VEC tmp, int psd);

void updatePseudoInverse1(int d, R_MAT pM, R_MAT M, R_VEC x,
		double alpha, R_VEC k, R_VEC u, R_VEC tmp, int psd);



#endif /* MYLIB_H_ */
