/*
 * mylib.c
 *
 *  Created on: Apr 13, 2016
 *      Author: kunkun220189
 */

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>


#include "mylib.h"
#include "nr.h"

double fabs(double x){ return x >= 0 ? x : 0; }
double sqr(double x){ return x*x;  }
double min(double x, double y){ return x < y ? x:y;}
double max(double x, double y){ return x >= y ? x:y;}

int cmpfunc (const void * a, const void * b){
	double diff;
	diff = ((entry *)a)->v - ((entry *)b)->v;
    return diff < 0 ? -1 : (diff == 0) ? 0 : 1;
}

/*---------------------------------------------------------------------*/
I_MAT createIM(int n, int m){
	register int i, j;
	I_MAT A = (int**) malloc(n * sizeof(int*));
	for( i = 0; i < n; ++ i){
		A[i] = (int*) malloc(m * sizeof(int));
		for( j = 0; j < m; ++ j)
			A[i][j] = 0;
	}
	return A;
}

I_VEC createIV(int n){
	register int i;
	I_VEC A = (int*) malloc (n * sizeof(int));
	for( i = 0; i < n; ++ i)
		A[i] = 0;
	return A;
}

R_MAT createRM(int n, int m){
	register int i, j;
	R_MAT A = (double**) malloc(n * sizeof(double*));
	for( i = 0; i < n; ++ i){
		A[i] = (double*) malloc(m * sizeof(double));
		for( j = 0; j < m; ++ j)
			A[i][j] = 0;
	}
	return A;
}

R_VEC createRV(int n){
	register int i;
	R_VEC A = (double*) malloc (n * sizeof(double));
	for( i = 0; i < n; ++ i)
		A[i] = 0;
	return A;
}

void destroyRM(R_MAT A, int n){
	register int i;
	for( i = 0; i < n; ++ i)
		free(A[i]);
}
void destroyIM(I_MAT A, int n){
	register int i;
	for( i = 0; i < n; ++ i)
		free(A[i]);
}

void  subtractVV(int d, R_VEC a, R_VEC b, R_VEC ret){
    register int i;
    for (i = 0; i < d; ++ i)
        ret[i] = a[i] - b[i];
}

/* ret = M*a*/
void multMV(int d, double** M, R_VEC a, R_VEC ret){
   register int i, j;
   for (i = 0; i < d; ++ i)
       for (ret[i] = 0, j = 0; j < d; ++ j)
           ret[i] += M[i][j] * a[j];
}

double multVV(int d, R_VEC a, R_VEC b){
	register int i;
	double ret = 0;
	for (i = 0; i < d; ++ i)
	   ret += a[i]*b[i];
	return ret;
}

void multVS(int d, R_VEC a, double alpha){
    register int i;
    for(i = 0; i < d; ++ i)
        a[i] *= alpha;
}
/*M = M + alpha*b*b' */
void multMSV(int d, R_MAT M, R_VEC b, double alpha){
    register int i, j;
    for( i = 0; i < d; ++ i)
        for (j = 0; j < d; ++ j)
            M[i][j] += alpha * b[i] * b[j];
}
/* M = M*alpha */
void multMS(int d, R_MAT M, double alpha){
    register int i, j;
    for( i = 0; i < d; ++ i)
        for (j = 0; j < d; ++ j)
            M[i][j] *= alpha;
}

/* get trace of matrix M */
double getTrace(int d, R_MAT M){
    double ret = 0;
    register int i;
    for( i = 0; i < d; ++ i)
        ret += M[i][i];
    return ret;
}

/* (a-b)'*M*(a-b) */
double distanceVV(int d, R_MAT M, R_VEC a, R_VEC b){
    double ret=0;
    register int i, j;
    if(M == NULL){
    	for (i = 0; i < d; ++i)
    	    ret += (a[i] - b[i])*(a[i] - b[i]);
    }else{
        for (i = 0; i < d; ++i)
            for(j = i; j < d; ++j)
            	ret +=(i == j?1:2) * M[i][j] * (a[i] - b[i]) * (a[j] - b[j]);
    }
    return ret;
}

/* ret = sqrt(a'*a) */
double norm(R_VEC a, int d){
    return sqrt(multVV(d, a, a));
}

/* random shuffle a vector u*/
void randperm(int n, I_VEC u){
    register int i;
    int randIdx, tmp;
    for (i = n-1; i; -- i){
        randIdx = rand() % (i+1);
        tmp = u[i], u[i]= u[randIdx], u[randIdx] = tmp;
    }
}
/*---------------------------------------------------------------------*/

double getDotProduct(pData input, int i, int j){
    if (getKernel(input, i, j) == -1){
        input->kernel[i * input->n + j] = multVV(input->d, getData(input, i), getData(input,j));
        input->kernel[j * input->n + i] = getKernel(input, i, j);
    }
    return getKernel(input, i, j);
}

/* update the similarity:
 * if   M == NULL, then sim[i] = sim[i] * alpha
 * if   M != NULL, then sim[i] = sim[i] + alpha * (u'*x[i])^2
 */
void updateStructure(pData input, R_MAT M, R_VEC u, double alpha){
	register int i;
	for( i = 0; i < input->n; ++ i){
		if( M != NULL )
			input->sim[i] += alpha * sqr(multVV(input->d, u, getData(input, i)));
		else
			input->sim[i] *= alpha;
	}
}

/**
 * M + alpha*(x[u] - x[v])*(x[u] - x[v])'
 */
void fastUpdateStructure(pData input, R_MAT M, int u, int v, double alpha){
	register int i;
	for( i = 0; i < input->n; ++ i)
		input->sim[i] += alpha *
               sqr( getDotProduct(input, u, i) - getDotProduct(input, v, i));
}

pData initialData(R_VEC X, I_VEC Y, int d, int n, int nClass,
        int k1, I_VEC pT, int k2, I_VEC pI){

	pData ret;
	register int i, cl;
	R_VEC v;

	ret    		= (pData) malloc (sizeof(struct Data));
	ret->d 		= d;
	ret->n 		= n;
	ret->nClass = nClass;
	ret->k1     = k1;
	ret->k2     = k2;

	ret->X 		= X;
	ret->Y 		= Y;
	ret->Targets= pT;
	ret->Impostors = pI;

	ret->temp   = (double *) malloc((n + 1) * sizeof(double));

	if (k2 < 0){ /* it is not a full computation */
		ret->sim    = (double *) malloc(n * sizeof(double));
		ret->NegNeighbors = createIM(nClass, n);
		for(i = 0; i < n; ++ i){
			cl = Y[i];
			ret->NegNeighbors[cl][0] ++ ;
			ret->NegNeighbors[cl][ret->NegNeighbors[cl][0]] = i;
		}
	    ret->kernel  = (double *) malloc(n * n * sizeof(double));
	    for(i = 0; i < n * n; ++ i)
	        ret->kernel[i] = -1;

	    for (i = 0; i < n; ++ i){
	    	v  = getData(ret, i);
	        ret->kernel[i * n + i] = multVV(d, v, v);
	        ret->sim[i] = ret->kernel[i * n + i];
	    }
	}

	return ret;
}

R_VEC getData(pData input, int index){
	return input->X + (index * input->d);
}

I_VEC getTargets(pData input, int index){
	return input->Targets + (index * input->k1);
}

I_VEC getImpostors(pData input, int index){
	return input->Impostors + (index * input->k2);
}

double getKernel(pData input, int i, int j){
	return input->kernel[i * input->n + j];
}

int getFarthestTarget(R_MAT M, pData input, int index, double* ret){

	register int i;
    int j = -1, b;
    double dist, tmp;
    I_VEC u = getTargets(input,index);

    *ret = -INF;

    if (input->k2 >= 0){
    	for(i = 0; i < input->k1; ++ i){
    		b    = u[i] - 1;
			dist = distanceVV(input->d, M, getData(input, index), getData(input, b));
			if (dist > *ret){
				*ret = dist;
				j     = b;
			}
    	}
    }else{
		multMV(input->d, M, getData(input, index), input->temp);
        
		for(i = 0; i < input->k1; ++ i){
			b     = u[i] - 1;

			dist  = input->sim[index] + input->sim[b];
			dist -= 2 * multVV(input->d, input->temp, getData(input, b));

			if (DEBUG){
				tmp = distanceVV(input->d, M, getData(input, index), getData(input, b));
				if (fabs(dist - tmp) > EPS){
					mexPrintf("A=%.10f B=%.10f A-B=%.10f\n", dist, tmp, dist - tmp);
					mexErrMsgTxt("ERROR of finding Targets\n");
				}
			}

			if (dist > *ret){
				*ret = dist;
				j     = b;
			}
		}
    }

    *ret = sqrt(max(0,*ret));
    
    return j;
}

int getNearestImpostor(R_MAT M, pData input, int index, double *ret){
	register int j;
	int c, l, b;
	double dist, tmp;
	I_VEC u;

	*ret = INF;

	if (input->k2 >= 0){
        u = getImpostors(input,index);
    	for(j = 0; j < input->k2; ++ j){
    		b    = u[j] - 1;
			dist = distanceVV(input->d, M, getData(input, index), getData(input, b));
			if (dist < *ret){
				*ret = dist;
				l    = b;
			}
    	}
	}else{
		multMV(input->d, M, getData(input, index), input->temp);

		for (c = 0; c < input->nClass; ++ c){

			if (c == input->Y[index])
				continue;

			for (j = 1; j <= input->NegNeighbors[c][0]; ++ j){
				b     = input->NegNeighbors[c][j];

				dist  = input->sim[index] + input->sim[b];
				dist -= 2 * multVV(input->d, input->temp, getData(input, b));

				if (DEBUG){
					tmp = distanceVV(input->d, M, getData(input, index), getData(input, b));
					if (fabs(dist - tmp) > EPS){
						mexPrintf("A=%.10f B=%.10f A-B=%.10f\n", dist, tmp, dist - tmp);
						mexErrMsgTxt("ERROR of finding Impostor\n");
					}
				}

				if (dist < *ret){
					*ret = dist;
					l    = b;
				}
			}
		}
	}

    *ret = sqrt(max(0,*ret));

	return l;
}

double getCost(R_MAT M, pData input){
	register int i;
	double ret = 0, dj, dl;

	for( i = 0; i < input->n; ++ i){
		getNearestImpostor(M, input, i, &dl);
		getFarthestTarget(M, input, i, &dj);
        dj = sqr(dj); /* */
        dl = sqr(dl); /* */
		ret += (dj + 1 - dl <= 0 ? 0 : dj + 1 - dl);
	}

	return ret;
}

void clean(pData input){
    if (input->k2 < 0){
	    destroyIM(input->NegNeighbors, input->nClass);
	    free(input->kernel);
        free(input->sim);
    }
    free(input->temp);
    free(input);
}

/****************************************************************************/
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
int lanczos(int d, double** A, int k, double *alpha, double* beta, double** Q, double *v, double *temp){
	int i, j, l;
	double nom, aux;

	for(i = 0; i < d; ++ i) v[i] = (double)rand()/RAND_MAX;
	nom = norm(v, d);
	for(i = 0; i < d; ++ i)
		Q[i][0] = v[i] / nom;

	for(j = 0; j < k; ++ j){
		for(i = 0; i < d; ++ i)
			for(v[i] = 0, l = 0; l < d; ++ l)
				v[i] += A[i][l]*Q[l][j];

		for(alpha[j] = 0, i = 0; i < d; ++ i)
			alpha[j] += Q[i][j] * v[i];

		for(i = 0; i < d; ++i)
			v[i] -= alpha[j] * Q[i][j];

		if(j > 0)
			for(i = 0; i < d; ++i)
                v[i] = v[i] - beta[j-1] * Q[i][j - 1];

		if (j < k - 1){

            for(i = 0; i <= j; ++ i){
                temp[i] = 0;
                for(l = 0; l < d; ++l)
                    temp[i] += Q[l][i] * v[l];
            }
            for(i = 0; i < d; ++ i) {
                aux = 0;
                for(l = 0; l <= j; ++ l)
                    aux += Q[i][l] * temp[l];
                v[i] -= aux;
            }

            beta[j] = norm(v, d);
            if ( fabs(beta[j]) <= 1e-13)
                return j + 1;
			for(i = 0; i < d; ++ i)
				Q[i][j+1] = v[i]/beta[j];
        }
	}

    return k;
}
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
        double * beta, double**V, double**z, double * temp, double ** X){

	register int i, j;
	int best = -1;
	double ret = -INF;

	for(i = 0; i < d; ++ i)
		for(j = 0; j < d; ++ j)
			X[i][j] = (i == j ? BIG_INT:0) - A[i][j];

	k = lanczos(d, X, k, alpha + 1, beta + 2, V, v, temp);

	for (i = 0; i <= k; ++ i){
		for(j = 0; j <= k; ++ j) z[i][j] = 0;
		z[i][i] = 1;
	}
	tqli(alpha, beta, k, z);
	for( i = 0; i < k; ++ i)
		if (ret < alpha[i + 1]){
			ret  = alpha[i + 1];
			best = i + 1;
		}
	for(i = 0; i < d; ++ i){
		v[i] = 0;
		for(j = 0; j < k; ++ j)
			v[i] += V[i][j] * z[j + 1][best];
	}
	return BIG_INT - ret;
}

/* a'*M*b */
double multTriplet(int d, R_MAT M, R_VEC a, R_VEC b){
    register int i, j;
    double ret = 0;
    for (i = 0; i < d; ++ i)
        for (j = 0; j < d; ++ j)
            ret += M[i][j] * a[i] * b[j];
    return ret;
}


int checkCondition_1(int d, R_MAT pM, R_MAT M,
		R_VEC a, R_VEC u, double nu, R_VEC tmp){
    double t, ret, aux;
    register int i, j;
    t   = multVV(d, a, u)/nu;
    multMV(d, pM, a, tmp);
    ret = 0;
    for(i = 0; i < d && ret <= EPS; ++ i){
        aux = 0;
        for (j = 0; j < d; ++ j)
            aux += M[i][j] * tmp[j];
        aux = a[i] - aux - t*u[i];
        ret += aux * aux;
    }
    return i == d;
}

int checkCondition_2(int d, R_MAT pM, R_MAT M, R_VEC a, R_VEC tmp){
    double ret, aux;
    register int i, j;
    multMV(d, pM, a, tmp);
    ret = 0;
    for(i = 0; i < d && ret <= EPS; ++ i){
        aux = 0;
        for (j = 0; j < d; ++ j)
            aux += M[i][j] * tmp[j];
        aux = a[i] - aux;
        ret+= aux * aux;
    }
    return i == d;
}


/* compute the parameter alpha*/
double learnAlpha(int d, R_MAT pM, R_MAT M, R_VEC a, R_VEC b,
		R_VEC k, R_VEC u, R_VEC tmp){
    double alpha, beta, nu, up, down, t, B, C, delta;
    alpha = 0;
    /*compute k*/
    multMV(d, pM, b, k);
    /*compute beta */
    beta  = 1 + multVV(d, k, b);

    /*compute u */
    multMV(d, M, k, tmp);
    subtractVV(d, b, tmp, u);
    /* norm u */
    nu = multVV(d, u, u);

    if ( fabs(nu) > EPS){
        /*Check condition*/
        if (checkCondition_1(d, pM, M, a, u, nu, tmp)){
            up = 1 - sqr(multVV(d, a, u)/nu);
            if ( up > EPS){
                down  = multTriplet(d, pM,a,a);
                down -= 2*multVV(d, a, k) * multVV(d, a, u)/nu;
                down += sqr(multVV(d, a, u)/nu) * (beta - 1);
                if (down > EPS)
                    alpha = up / down;
                else
                    alpha = INF;
            }
        }
    }else{
        if (checkCondition_2(d, pM, M, a, tmp)){
            t = multTriplet(d, pM, a, a);
            B = beta - 1 - t;
            C = max(0.0, (beta - 1)*t - sqr(multVV(d, k, a)));
            delta = sqr(B) + 4*C;
            t = -B + sqrt(delta);
            if (t <= EPS)
                alpha = INF;
            else
                alpha = 2/t;
        }
    }

    return (alpha<0? 0:alpha);
}

void updatePseudoInverse(int d, R_MAT pM, R_MAT M, R_VEC x,
		double alpha, R_VEC k, R_VEC u, R_VEC tmp){
    double beta, nu, aux, nk;
    register int i, j;

    /*compute k*/
    multMV(d, pM, x, k);
    /*compute beta */
    beta  = 1 + multVV(d, k, x);

    /*compute u*/
    multMV(d, M, k, tmp);
    subtractVV(d, x, tmp, u);
    /* norm u*/
    nu = multVV(d, u, u);
    if ( fabs(nu) > EPS){
        aux = beta - 1 + 1/alpha;
        for (i = 0; i < d; ++ i)
            for( j = 0; j < d; ++ j){
                pM[i][j] -= k[i] * u[j] / nu;
                pM[i][j] -= k[j] * u[i] / nu;
                pM[i][j] += aux  * u[i] * u[j] / sqr(nu);
            }
    }else{
        if (fabs(1 + (beta - 1)*alpha ) > EPS){
            aux = alpha/((beta - 1)*alpha + 1);
            for (i = 0; i < d; ++ i)
                for( j = 0; j < d; ++ j)
                    pM[i][j] -= aux * k[i]*k[j];
        }else{
            nk = multVV(d, k, k);
            multMV(d, pM, k, tmp);
            aux = multVV(d, tmp, k) / sqr(nk);
            for (i = 0; i < d; ++ i)
                for( j = 0; j < d; ++ j){
                    pM[i][j] -= k[i] * tmp[j] / nk;
                    pM[i][j] -= k[j] * tmp[i] / nk;
                    pM[i][j] += aux  * k[i] * k[j];
                }
        }
    }
}


/*-----------------------------------------------------------------------*/
/* compute the parameter alpha*/
double learnAlpha1(int d, R_MAT pM, R_MAT M, R_VEC a, R_VEC b,
		R_VEC k, R_VEC u, R_VEC tmp, int psd){
    double alpha, beta, nu, up, down, t, B, C, delta;
    alpha = 0;
    /*compute k*/
    multMV(d, pM, b, k);
    /*compute beta */
    beta  = 1 + multVV(d, k, b);
    if (psd) goto no_check;
    /*compute u */
    multMV(d, M, k, tmp);
    subtractVV(d, b, tmp, u);
    /* norm u */
    nu = multVV(d, u, u);

    if ( fabs(nu) > EPS){
        /*Check condition*/
        if (checkCondition_1(d, pM, M, a, u, nu, tmp)){
            up = 1 - sqr(multVV(d, a, u)/nu);
            if ( up > EPS){
                down  = multTriplet(d, pM,a,a);
                down -= 2*multVV(d, a, k) * multVV(d, a, u)/nu;
                down += sqr(multVV(d, a, u)/nu) * (beta - 1);
                if (down > EPS)
                    alpha = up / down;
                else
                    alpha = INF;
            }
        }
    }else{
        if (checkCondition_2(d, pM, M, a, tmp)){
            no_check:
            t = multTriplet(d, pM, a, a);
            B = beta - 1 - t;
            C = max(0.0, (beta - 1)*t - sqr(multVV(d, k, a)));
            delta = sqr(B) + 4*C;
            t = -B + sqrt(delta);
            if (t <= EPS)
                alpha = INF;
            else
                alpha = 2/t;
        }
    }

    return (alpha<0? 0:alpha);
}


void updatePseudoInverse1(int d, R_MAT pM, R_MAT M, R_VEC x,
		double alpha, R_VEC k, R_VEC u, R_VEC tmp, int psd){
    double beta, nu, aux, nk;
    register int i, j;

    /*compute k*/
    multMV(d, pM, x, k);
    /*compute beta */
    beta  = 1 + multVV(d, k, x);
    if (psd) goto label;
    /*compute u*/
    multMV(d, M, k, tmp);
    subtractVV(d, x, tmp, u);
    /* norm u*/
    nu = multVV(d, u, u);
    if ( fabs(nu) > EPS){
        aux = beta - 1 + 1/alpha;
        for (i = 0; i < d; ++ i)
            for( j = 0; j < d; ++ j){
                pM[i][j] -= k[i] * u[j] / nu;
                pM[i][j] -= k[j] * u[i] / nu;
                pM[i][j] += aux  * u[i] * u[j] / sqr(nu);
            }
    }else{
        if (fabs(1 + (beta - 1)*alpha ) > EPS){
            label:
            aux = alpha/((beta - 1)*alpha + 1);
            for (i = 0; i < d; ++ i)
                for( j = 0; j < d; ++ j)
                    pM[i][j] -= aux * k[i]*k[j];
        }else{
            nk = multVV(d, k, k);
            multMV(d, pM, k, tmp);
            aux = multVV(d, tmp, k) / sqr(nk);
            for (i = 0; i < d; ++ i)
                for( j = 0; j < d; ++ j){
                    pM[i][j] -= k[i] * tmp[j] / nk;
                    pM[i][j] -= k[j] * tmp[i] / nk;
                    pM[i][j] += aux  * k[i] * k[j];
                }
        }
    }
}

/*------------------------------------------------------------------------*/

double trainingAccuracy(pData input, R_MAT M){
	int i, j, l, ind, rep, correct;
	entry * dist = (entry*) malloc(input->n * sizeof (entry));
	int * cnt = (int*) malloc(input->nClass * sizeof(int));

	for (i = 0, correct = 0; i < input->n; ++ i){

		for (j = 0, l = 0; j < input->n; ++ j){
			if (i == j) continue;
			dist[l].ind = j;
			dist[l].v   = distanceVV(input->d, M, getData(input, i), getData(input,j));
			++ l;
		}

		for (j = 0; j < input->nClass; ++ j)
			cnt[j] = 0;

		qsort(dist, l, sizeof(entry), cmpfunc);
		for (j = 0; j < input->k1; ++ j){
			ind = dist[j].ind;
			++ cnt[input->Y[ind]];
		}

		ind = -1;
		rep = -1;

		for (j = 0; j < input->nClass; ++ j){
			if (cnt[j] > rep){
				rep = cnt[j];
				ind = j;
			}
		}

		if (ind == input->Y[i])
			++ correct;
	}

    free(dist);
    free(cnt);

    return correct * 100.0 / input->n;
}



