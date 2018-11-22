#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <time.h>

#include "mylib.h"
#include "mex.h"

#define EXACTINPUTS  8
#define EXACTOUTPUTS 2


/* Input Arguments */
#define	INPUT_X         prhs[0] /* input pseudo inverse matrix */
#define INPUT_Y         prhs[1] /* label matrix  */
#define	INPUT_T         prhs[2] /* target matrix  */
#define INPUT_nCls      prhs[3] /* number of different class */
#define	INPUT_c  	    prhs[4] /* hyperparameter */
#define	INPUT_lr  	    prhs[5] /* learning rate  */
#define INPUT_a         prhs[6] /* is approximate? */
#define INPUT_MAX_ITER  prhs[7] /* maximum of iterations */
#define INPUT_I         prhs[8] /* impostor matrix */

/* Output Arguments */
#define	OUTPUT_M    	plhs[0] /* updated pseudoinverse matrix */
#define	OUTPUT_C    	plhs[1] /* cost */

/*------------------------------------------------------------------------*/

FILE * fp;

double distanceMetricLearning(pData input, double c, double lr, 
        double **M, int MAX_ITERS){
    
    if (DEBUG){ 
       fp = fopen ("file.txt", "w");
    }
   
    R_VEC First, Second;
    I_VEC order;

    R_VEC  temp, alpha, beta, v;
    R_MAT  Q, Z, tempM;

    double dj, dl, cost, eta;
    int i, j, l, t, k;
    int update, iters, epoch;
    int v_index, tmp;

    int d = input->d;
    int n = input->n;

    /* initial parameters */
    k = MAX_LACZOS;
    if ( d <= MAX_LACZOS )
    	k = d < 10 ? d : 10;

    order    = createIV(n);
    temp     = createRV(d);
    First    = createRV(d);
    Second   = createRV(d);
    v        = createRV(d);
    alpha    = createRV(k + 1);
    beta     = createRV(k + 1);
    Q 	     = createRM(d, d);
    Z 	     = createRM(k + 1, k + 1);
    tempM    = createRM(d, d);

    /* the initial solution */
    for (i = 0; i < d; ++ i)
    	M[i][i]  = c/d;
    
    if (input->k2 < 0)
        updateStructure(input, NULL, NULL, c/d);

    for (i = 0; i < n; ++ i)
    	order[i] = i;

    /* random the order of stochastic gradient */
    randperm(n, order);

    /* begin algorithm */        
    update = n;
    epoch  = 1;
    iters  = 1;
    
    for ( ;(update > 0.01 * n) && (epoch <= MAX_EPOCH); ++ epoch){


    	if (DEBUG){
            mexPrintf("#Iter=%d, Cost=%.10f\n", iters, getCost(M, input));
            fprintf(fp, "%.1f, %.5f, %.2f\n", epoch-1.0, getCost(M, input), 
                         trainingAccuracy(input,M));
        }
        
        /* run new epoch */
        cost    = 0;
        update  = 0;        
        v_index = 0; /* index of the violated element in the last iteration */

        for (t = 0; (t < n) && (iters <= MAX_ITERS); ++ t, ++ iters){

            i = order[t];
            j = getFarthestTarget (M, input, i, &dj);
            l = getNearestImpostor(M, input, i, &dl);
            
            dj = sqr(dj); /* */
            dl = sqr(dl); /* */
            
            /* if it is a violated example */
            if (dj + 1 > dl && dj + dl > 2*EPS){
                cost += dj + 1 - dl;

                /* finding vector First and Second */
                subtractVV(d, getData(input, j), getData(input, i) , First);
                subtractVV(d, getData(input, l), getData(input, i) , Second);

                /* step size */
                eta = lr/sqrt(iters);
                
                /* if there exists the gradient */
                if (dl > EPS){
                    multMSV(d, M, Second, eta);
                    if (input->k2 < 0)
                        fastUpdateStructure(input, M, i, l, eta);
                }

                /* if there exists the gradient */
                if (dj > EPS){
                    multMSV(d, M, First, -eta);
                    if (input->k2 < 0)
                        fastUpdateStructure(input, M, i, j, -eta);
                }
                
                /* finding the smallest eigenvalue */
                eta = getSmallestEigenvalue(k,d,M,v,alpha,beta,Q,Z,temp,tempM);
                
                /* if the matrix contains a negative eigenvalue */
                if (eta < 0){
                    multMSV(d, M, v, -eta);
                    if (input->k2 < 0)
                        updateStructure(input, M, v, -eta);
                }

                /* we need to truncate the solution to satisfy the trace bound */
                eta = getTrace(d, M);
                if (eta > c){
                    multMS(d, M, c/eta);
                    if (input->k2 < 0)
                        updateStructure(input, NULL, NULL, c/eta);
                }

                update ++;
                /* change order for the next iteration */
                tmp            = order[t];
                order[t]       = order[v_index];
                order[v_index] = tmp;
                v_index ++;
            }
            
            if (DEBUG && t == n/2){                
                fprintf(fp, "%.1f, %.5f, %.2f\n", epoch-0.5, getCost(M, input), 
                         trainingAccuracy(input,M));
            }
        }
        
        mexPrintf("Update (%d) = %.2f%c\n", update, update*100.0/n, '%');
    }
    
    if (DEBUG){
        cost = getCost(M, input);
        fprintf(fp, "%d, %.5f, %.2f\n", epoch-1, cost, trainingAccuracy(input,M));        
        fclose(fp);
    }
    
    /* release memory */
    free(temp);
    free(First);
    free(Second);
    free(order);
    free(alpha);
    free(beta);
    free(v);
    destroyRM(Z, k + 1);
    destroyRM(Q, d);
    destroyRM(tempM, d);
    
    clean(input);    
    
    return cost;
}


double approximateDistanceMetricLearning(pData input, double c, double lr, 
        double **M, int MAX_ITERS){
    
    if (DEBUG)
        fp = fopen ("file.txt", "w");

	R_MAT pM;
	R_VEC First, Second, k, u;
	I_VEC order;

	double dj, dl, alpha, cost;
	int update, iters, epoch, psd;
    int v_index, tmp;

	register int i, j, l, t;

	int d = input->d;
	int n = input->n;

	int nIters = min(MAX_EPOCH * n, MAX_ITERS);
	double reg;
    
	pM     = createRM(d, d);
	First  = createRV(d);
	Second = createRV(d);
	k      = createRV(d);
	u      = createRV(d);
	order  = createIV(n);

	/* the initial solution */
	for (i = 0; i < d; ++ i){
		M [i][i] = c/d;
		pM[i][i] = d/c;
	}

	if (input->k2 < 0)
		updateStructure(input, NULL, NULL, c/d);

	for (i = 0; i < n; ++ i)
		order[i] = i;

    randperm(n, order);
	/* begin algorithm */
	iters  = 1;
	epoch  = 1;
	update = n;

	for (; (update > 0.01 * n) && (epoch <= MAX_EPOCH); ++ epoch){

        if (DEBUG){
            mexPrintf("#Iter=%d, Cost=%.10f\n", iters, getCost(M, input));
            fprintf(fp, "%.1f, %.5f, %.2f\n", epoch-1.0, getCost(M, input), 
                         trainingAccuracy(input,M));
        }
        
        /* run new epoch */
        update  = 0;
        cost    = 0;
        v_index = 0;
        
		for (t = 0; t < n && iters < MAX_ITERS; ++ t, ++ iters){

            i = order[t];
            j = getFarthestTarget (M, input, i, &dj);
            l = getNearestImpostor(M, input, i, &dl);
            
            dj = sqr(dj); /* */
            dl = sqr(dl); /* */
                    
			if (dj + 1 > dl){
                cost += dj + 1 - dl;

                /* finding vector First and Second*/
                subtractVV(d, getData(input, j), getData(input, i), First);
                subtractVV(d, getData(input, l), getData(input, i), Second);

                /*multVS(d, First,  dj > EPS ? sqrt(0.5/dj) : 0); */ /* */
                /*multVS(d, Second, dl > EPS ? sqrt(0.5/dl) : 0); */ /* */

                psd = (iters <= nIters - 100);
                reg = psd ? 0.9 : 1;
                
              /*alpha = learnAlpha(d, pM, M, First, Second, k, u, input->temp);*/
                alpha = learnAlpha1(d, pM, M, First, Second, k, u, input->temp, psd);                
                alpha = min(reg * alpha, lr/sqrt(iters));

               if (alpha > EPS){
                    updatePseudoInverse1(d, pM, M, Second, alpha, k, u, input->temp, psd);                       
                    multMSV(d, M, Second, alpha);
                	if (input->k2 < 0)
                		fastUpdateStructure(input, M, i, l, alpha);

                    updatePseudoInverse1(d, pM, M, First,-alpha, k, u, input->temp, psd);                        
                    multMSV(d, M, First, -alpha);
                	if (input->k2 < 0)
                		fastUpdateStructure(input, M, i, j, -alpha);
                    
					alpha = getTrace(d, M);
					if (alpha > c){
						alpha = c / alpha;
						multMS(d, M, alpha);
						multMS(d, pM, 1/alpha);
						if (input->k2 < 0)
							updateStructure(input, NULL, NULL, alpha);
					}
                    
					++ update;                    
                    /* change order for the next iteration */
                    tmp            = order[t];
                    order[t]       = order[v_index];
                    order[v_index] = tmp;
                    v_index ++;
				}
			}
            if (DEBUG && (t == n/2)){
                 fprintf(fp, "%.1f, %.5f, %.2f\n", epoch-0.5, getCost(M, input), 
                         trainingAccuracy(input,M));
            }
		}
        
        mexPrintf("Update (%d) = %.2f%c\n", update, update*100.0/n, '%');
	}
    
    if (DEBUG){
        cost = getCost(M, input);
        fprintf(fp, "%d, %.5f, %.2f\n", epoch-1, cost, trainingAccuracy(input,M));        
        fclose(fp);
    }
    
    
	/* release memory */
	free(First);
	free(Second);
	free(k);
	free(u);
	free(order);
	destroyRM(pM, d);
    
    clean(input);
    
	return cost;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{

    R_VEC pX, pM;     
    I_VEC pT, pY, pI = NULL;
    R_MAT M;
    
    pData input;
    double c, lr;   
    int d, n, k1 = -1, k2 = -1, nClass;
    int i, j;
    int approx, m_iters;
    
    if (nrhs < EXACTINPUTS) {
        mexPrintf("Check input parameters\n");    
        return;
    }
    
    if (nrhs > EXACTINPUTS){
        if (DEBUG) mexPrintf("Using pre impostor\n");
    	pI = (I_VEC) mxGetPr(INPUT_I);
    	k2 = mxGetM(INPUT_I);     
    }
    
    /* get inputs */
    pX     = (R_VEC) mxGetPr(INPUT_X);
    pT     = (I_VEC) mxGetPr(INPUT_T);
    pY     = (I_VEC) mxGetPr(INPUT_Y);        
    d      = mxGetM(INPUT_X);
    n      = mxGetN(INPUT_X);
    k1     = mxGetM(INPUT_T);
    c      = mxGetScalar(INPUT_c);
    lr     = mxGetScalar(INPUT_lr);
    nClass = (int)(mxGetScalar(INPUT_nCls) + 1e-9);
    approx = (int)(mxGetScalar(INPUT_a) + 1e-9);
    m_iters= (int)(mxGetScalar(INPUT_MAX_ITER) + 1e-9);
    
    /* build output configure */
    OUTPUT_C = mxCreateDoubleMatrix(1, 1, mxREAL);
    OUTPUT_M = mxCreateDoubleMatrix(d, d, mxREAL);
    pM       = (R_VEC)mxGetPr(OUTPUT_M);
    
    input    = initialData(pX, pY, d, n, nClass, k1, pT, k2, pI);
    
    /* make dummy matrix for learning*/
    M = createRM(d, d);
    *mxGetPr(OUTPUT_C) = approx
                       ? approximateDistanceMetricLearning(input, c, lr, M, m_iters)
                       : distanceMetricLearning(input, c, lr, M, m_iters);
                             
    for (i = 0; i < d; ++ i)
        for(j = 0; j < d; ++ j)
            pM[i*d+j] = M[i][j];
    /* free memory */
    destroyRM(M, d);
}







