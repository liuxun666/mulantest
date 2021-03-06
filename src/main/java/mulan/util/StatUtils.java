/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package mulan.util;


import mulan.rbms.M;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

/**
 * StatUtils - Helpful statistical functions.
 * @author Jesse Read (jesse@tsc.uc3m.es)
 */
public abstract class StatUtils {

	//
	// EMPIRICAL DISTRIBUTIONS
	//
	public static double[][] Y;
	/**
	 * P - Empirical prior.
	 * @param	Y[][]	label matrix
	 * @param	x	 	label values
	 * @return 	[P(Y_1==x[1]), P(Y_2==x[2]), ..., P(Y_L==x[L])]
	 */
	public static double[] P(double Y[][], int x[]) {
		int L = x.length;
		return P(Y,MLUtils.gen_indices(L),x);
	}

	/**
	 * P - Empirical prior.
	 * @param	Y[][]	label matrix
	 * @param	x	 	label values
	 * @param	j		label indices
	 * @return 	[P(Y_j[1]==x[1]), P(Y_j[2]==x[2]), ..., P(Y_j[L]==x[L])]
	 */
	public static double[] P(double Y[][], int j[], int x[]) {
		int L = j.length;
		double p[] = new double[L];
		for(int j_ = 0; j_ < L; j_++) {
			p[j_] = p(Y,j[j_],x[j_]);
		}
		return p;
	}

	/**
	 * p - Empirical prior.
	 * In the multi-label case, k in {0,1}
	 * @param	Y[][]	label matrix
	 * @param	j		label index
	 * @param	x	 	label value
	 * @return 	P(Y_j==k) in Y.
	 */
	//p(double[][]:标签的矩阵，int：标签索引号 ，int：标签值：0或1或-1)
	public static double p(double Y[][], int j, int k) {
		int N = Y.length;//N的值可能是L,即是标签的数目
		double p = 0.0001;
		for(int i = 0; i < N; i++) {
			if ((int)Math.round(Y[i][j]) == k) {
				p += 1.0;
			}
		}
		return p/N;
	}

	public static double p(double Y[][], int j, double k) {
		int N = Y.length;//N的值可能是L,即是标签的数目
		double p = 0.0001;
		for(int i = 0; i < N; i++) {
			if ((int)Math.round(Y[i][j]) == (int)Math.round(k)) {
				p += 1.0;
			}
		}
		return p/N;
	}

	/**
	 * p - Empirical prior.
	 * In the multi-label case, k in {0,1}
	 * @param	D    	Instances
	 * @param	j		label index
	 * @param	x	 	label value
	 * @return 	P(Y_j==k) in D.
	 */
	public static double p(Instances D, int j, int k) {
		return p(MLUtils.getYfromD(D),j,k);
	}

	/**
	 * p - Empirical joint.
	 * Multi-target friendly.
	 * @param	Y[][]	label matrix
	 * @param	j		label index
	 * @param	x	 	label value
	 * @return 	P(Y_j = v, Y_k = w) in Y.
	 */
	public static double P(double Y[][], int j, int v, int k, int w) {
		int N = Y.length;
		double p = 0.0001;
		for(int i = 0; i < N; i++) {
			if (((int)Math.round(Y[i][j]) == v) && ((int)Math.round(Y[i][k]) == w))
				p += 1.0;
		}
		return p/N;
	}

	public static double P(double Y[][], int j, double v, int k, double w) {
		int N = Y.length;
		double p = 0.0001;
		for(int i = 0; i < N; i++) {
			if (((int)Math.round(Y[i][j]) == v) && ((int)Math.round(Y[i][k]) == w))
				p += 1.0;
		}
		return p/N;
	}

	/**
	 * p - Empirical joint.
	 * Multi-target friendly.
	 * @param	D       Instances
	 * @param	j		label index
	 * @param	x	 	label value
	 * @return 	P(Y_j = v, Y_k = w) in D.
	 */
	public static double P(Instances D, int j, int v, int k, int w) {
		return P(MLUtils.getYfromD(D),j,v,k,w);
	}

	/**
	 * Delta(x_1,x_2,x_3 = v_1,v_2,v_3) for j = 1,2,3, k = 1,2,3.
	 */
	private static boolean match(Instance x, int indices[], int values[]) {
		for(int j = 0; j < indices.length; j++) {
			int v = (int)Math.round(x.value(indices[j]));
			if (v != values[j]) {
				return false;
			}
		}
		return true;
	}
	
	/**
	 * P - Empirical joint.
	 * Multi-target friendly.
	 * @param	D       Instances
	 * @param	j[]		label indices, e.g., 1,2,3
	 * @param	v[]		label values, e.g., 0,0,1
	 * @return 	P(x_1,x_2,x_3 = v_1,v_2,v_3) for j = 1,2,3 in D
	 */
	public static double P(Instances D, int j[], int v[]) {
		int N = D.numInstances();
		int n = 0;
		for (Instance x : D) {
			if (match(x,j,v))
				n++;
		}
		return Math.max(0.0001,(double)n/N);
	}

	/**
	 * jPMF - Joint PMF.
	 * @return the joint PMF of the jth and kth labels in D.
	 */
	public static double[][] jPMF(Instances D, int j, int k) {
		double JOINT[][] = new double[D.attribute(j).numValues()][D.attribute(k).numValues()];
		int N = D.numInstances();
		for(int i = 0; i < N; i++) {
			int v_j = (int)Math.round(D.instance(i).value(j));
			int v_k = (int)Math.round(D.instance(i).value(k));
			JOINT[v_j][v_k] += (1.0 / (double)N);
		}
		return JOINT;
	}

	/**
	 * Joint Distribution.
	 * @return the joint PMF of the jth and kth and lthlabels in D.
	 */
	public static double[][][] jPMF(Instances D, int j, int k, int l) {
		double JOINT[][][] = new double[D.attribute(j).numValues()][D.attribute(k).numValues()][D.attribute(l).numValues()];
		int N = D.numInstances();
		for(int i = 0; i < N; i++) {
			int v_j = (int)Math.round(D.instance(i).value(j));
			int v_k = (int)Math.round(D.instance(i).value(k));
			int v_l = (int)Math.round(D.instance(i).value(l));
			JOINT[v_j][v_k][v_l] += (1.0 / (double)N);
		}
		return JOINT;
	}

	/**
	 * I - Information Gain. //信息增益
	 * Multi-target friendly (does not assume binary labels).
	 * @return I(Y_j;Y_k) in dataset D.
	 */
	public static double I(Instances D, int j, int k) {
		double I = 0.0;
		for(int x = 0; x < D.attribute(j).numValues(); x++) {
			double p_x = p(D,j,x);
			for(int y = 0; y < D.attribute(k).numValues(); y++) {
				double p_y = p(D,k,y);
				double p_xy = P(D,j,x,k,y);
				I += p_xy * Math.log ( p_xy / ( p_x * p_y) );
			}
		}
		//返回的是标签之间的信息增益
		return I;
	}

	public static double I(double[][] D, int j, int k) {
		double I = 0.0;

		for(int x = 0; x < D[j].length; x++) {
			double p_x = p(D,j, D[j][x]);
			for(int y = 0; y < D[k].length; y++) {
				double p_y = p(D,k,D[k][y]);
				double p_xy = P(D,j,D[j][x],k,D[k][y]);
				I += p_xy * Math.log ( p_xy / ( p_x * p_y) );
			}
		}
		//返回的是标签之间的信息增益
		return I;
	}

	/** Critical value used for Chi^2 test. */
	public static final double CRITICAL[] = new double[]{0.,2.706, 4.605, 6.251, 7.779};      // P == 0.10

	/**
	 * Chi^2 - Do the chi-squared test on the jth and kth labels in Y.
	 * @NOTE multi-label only! @TODO Use enumerateValues() !!!
	 * If they are correlated, this means unconditional dependence!
	 * @return	The chi-square statistic for labels j and k in Y.
	 */
	public static double chi2 (Instances Y,int j,int k) {
		// H_0 : p(Y_j,Y_k) = p(Y_j)p(Y_k)

		double chi2 = 0.0;
		for(int j_ = 0; j_ < 2; j_++) {
			for(int k_ = 0; k_ < 2; k_++) {
				double E = p(Y,j,j_) * p(Y,k,k_); 			// Expected vaule P(Y_j = j_)P(Y_k = k_)
				double O = P(Y,j,j_,k,k_);					// Observed value P(Y_j = j_, Y_k = k_)
				chi2 += ( ((O - E) * (O - E)) / E );
			}
		}
		return chi2;
	}

	/**
	 * Chi^2 - Chi-squared test.
	 * If they are correlated, this means unconditional dependence!
	 * @param	M[][][]			measured joint  P(Y_1,Y_2)      
	 * @param	Exp[][][]		expect joint 	P(Y_1)P(Y_2)	given null hypothesis
	 * @return	The chi-square statistic for labels j and k in Y; normalized by critical value.
	 */
	public static double[][] chi2 (double M[][][], double Exp[][][]) {

		int K = M.length;
		int L = M[0].length;
		int DoF = K - 1;

		double V[][] = new double[L][L];

		for(int i = 0; i < K; i++) {
			for(int j = 0; j < L; j++) {
				for(int k = j+1; k < L; k++) {
					double J = M[i][j][k];		// actual (joint) 			p(e==e)
					double E = Exp[i][j][k];	// expected (prior*prior)	
					V[j][k] += ( ((J - E) * (J - E)) / E );
				}
			}
		}
		//System.out.println(weka.core.M.toString((double[][])V));
		// offset
		double p = CRITICAL[DoF];
		for(int j = 0; j < L; j++) {
			for(int k = j+1; k < L; k++) {
				V[j][k] -= p;
			}
		}
		return V;
	}

	/**
	 * MargDepMatrix - Get an Unconditional Depndency Matrix.
	 * (Works for both ML and MT data).
	 * @param	D	dataset
	 * @return a L*L matrix representing Unconditional Depndence.
	 */
	public static double[][] margDepMatrix(Instances D) {
		return margDepMatrix(D,D.classIndex());
	}

	/**
	 * MargDepMatrix - Get an Unconditional Depndency Matrix.
	 * (Works for both ML and MT data).
	 * @param	D	dataset
	 * @param	L	number of labels
	 * @return a L*L matrix representing Unconditional Depndence.
	 */
	public static double[][] margDepMatrix(Instances D, int L) {
		double M[][] = new double[L][L];
		for(int j = 0; j < L; j++) {
			for(int k = j+1; k < L; k++) {
				// get I(Y_j;X_k)
				//I(intsances：D，int j，int k):就是计算信息增益的，
				//计算在数据集D上，I(Y_j;Y_k)标签Y_j与Y_k的信息增益
				M[j][k] = I(D,j,k);
			}
		}
		//M中存储的就是标签之间的信息增益
		return M;
	}

	public static double[][] margDepMatrix(double[][] D, int L) {
		int[][] D_ = new int[D.length][D[0].length];
		for (int i = 0; i < D.length; i++) {
			for (int j = 0; j < D[i].length; j++) {
				D_[i][j] = (int)D[i][j];
			}
		}
		return margDepMatrix(D_, L);
	}

	public static double[][] margDepMatrix(int[][] D, int L) {
		double M[][] = new double[L][L];
		for(int j = 0; j < L; j++) {
			for(int k = j + 1; k < L; k++) {
				M[j][k] = mi(D, j, k);
			}
		}
		//M中存储的就是标签之间的信息增益
		return M;
	}

	public static double[][] mInfomation(int[][] D, int L) {
		double[][] D_ = new double[D.length][D[0].length];
		for (int i = 0; i < D.length; i++) {
			for (int j = 0; j < D[i].length; j++) {
				D_[i][j] = (double)D[i][j];
			}
		}
		double M[][] = new double[L][L];
		for(int j = 0; j < L; j++) {
			for(int k = j + 1; k < L; k++) {
				M[j][k] = MutualInformation.calculateMutualInformation(D_[j], D_[k]);
			}
		}
		//M中存储的就是标签之间的信息增益
		return M;
	}

	public static double  mi(int[][] D, int idx1, int idx2) {
		double s = 0;
        double[][] tmp = new double[2][2];
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                tmp[i][j] = -1;
            }
        }

        for (int i = 0; i < 2; i++) {
            double px = Px_v(D, idx1, i);
            for (int j = 0; j < 2; j++) {
                double py = Px_v(D, idx2, j);
                double p_xy = Pxy(D, idx1, idx2, i, j);
                if(i == j){
					s += p_xy * (Math.log(p_xy / (px * py)) / Math.log(2.0));
				}else{
                	s -= p_xy * (Math.log(p_xy / (px * py)) / Math.log(2.0));
				}
            }
        }

//        for (int i = 0; i < D.length; i++) {
////            if(tmp[D[i][idx1]][D[i][idx2]] != -1){
////                s += tmp[D[i][idx1]][D[i][idx2]];
////            }else{
//                double p_xy = Pxy(D, idx1, idx2, D[i][idx1], D[i][idx2]);
//                double px = Px_v(D, idx1, D[i][idx1]);
//                double py = Px_v(D, idx2, D[i][idx2]);
//                double t = p_xy * Math.log(p_xy / (px * py));
//                s += t;
//                tmp[D[i][idx1]][D[i][idx2]] = t;
////            }
//
//        }

		return s;
	}


	public static double Pxy(int[][] D, int idx1, int idx2, int x, int y){
		double sum = 0.01;
		for (int i = 0; i < D.length; i++) {
			if(D[i][idx1] == x && D[i][idx2] == y){
				sum += 1;
			}
		}
		if(sum == 0.01) System.out.println(idx1 + " -> " + idx2 + " sum = 0");
		return sum / D.length;
	}

	public static double Px_v(int[][] D, int idx, int value){
		double sum = 0.01;
		for (int i = 0; i < D.length; i++) {
			if(D[i][idx] == value){
				sum += 1;
			}
		}
        if(sum == 0.01) System.out.println(idx  + " sum = 0");

        return sum / D.length;
	}


	// A bit of a useless function -- get rid of it somehow?
	private static double[] fillError(Result result, int L) {

		double Yprob[][] = result.allPredictions(); 
		int Ytrue[][] = result.allActuals(); 
		double ts[] = ThresholdUtils.thresholdStringToArray(result.getInfo("Threshold"),L); // <-- @TODO should not assume this for multi-target
		int Ypred[][] = ThresholdUtils.threshold(Yprob,ts);

		double E[] = new double[L];
		for(int j = 0; j < L; j++) {
			//E[j] = 1.0 - result.output.get("Accuracy["+j+"]");
			E[j] = Metrics.P_Hamming(Ytrue,Ypred,j);
		}
		return E;
	}

	/**
	 * CondDepMatrix - Get a Conditional Depndency Matrix.
	 * @version My version, based on Zhang's 'LEAD' approach:<br> 
	 * the probability of labels j and k both getting errors on the same instance is L_loss(j)*L_loss(k)
	 * if the actual co-occurence is otherwise. 
	 * @version note: currently we are only looking at two kinds: are the scores correlated or not
	 * @version H0: the correlated scores == score*score
	 * @param	D	dataset
	 * @return a L*L matrix of Unconditional Depndence.
	 */
	public static double[][] condDepMatrix(Instances D, Result result) {

		int L = D.classIndex();
		int N = D.numInstances();
		double T[][] = MLUtils.getYfromD(D);						// OUTPUT (TEACHER)
		double Y[][] = M.threshold(result.allPredictions(),0.5);	// OUTPUT (PREDICTED)
		result.output = Result.getStats(result,"6");	            // <-- high verbosity, because we need individual accuracies				
		double E[] = fillError(result, L);							// ERRORS (EXPECTED)
		double F[][][] = new double[3][L][L];						// ERRORS (ACTUAL)
		// Find the actual co-occurence ...
		for(int i = 0; i < N; i++) {
			int y[] = MLUtils.toIntArray(Y[i],0.5); 				// predicted
			int t[] = MLUtils.toIntArray(T[i],0.5);					// actual (teacher)
			for(int j = 0; j < L; j++) {
				for(int k = j+1; k < L; k++) {
					if (y[j] != t[j] && y[k] != t[k]) {
						// if j incorrect and k also ...
						F[0][j][k]++;								// error type 0
					}
					else if (y[j] == t[j] && t[k] == y[k]) {
						// both are correct
						F[2][j][k]++;								// error type 2
					}
					else {
						// if only one is correct
						F[1][j][k]++;								// error type 1
					}
				}
			}
		}

		// UnNormalize with the Expected error
		double E_norm[][][] = new double[3][L][L];
		for(int j = 0; j < L; j++) {
			for(int k = j+1; k < L; k++) {
				E_norm[0][j][k] = N * (E[j] * E[k]);
				E_norm[2][j][k] = N * ((1.0 - E[k]) * (1.0 - E[j]));
				E_norm[1][j][k] = N * ( (E[j] * (1.0 - E[k])) + (1.0 - E[j]) * E[k]);
			}
		}
		return StatUtils.chi2(F,E_norm);
	}

	/**
	 * LEAD. 
	 * Do the chi-squared LEAD test on all labels in D.
	 * 在数据集D上，对所有标签做卡方检验，卡方检验就是衡量标签实际值与理论值之间的符合程度
	 * We would expect the 3 kinds of error to be uncorrelacted.
	 * However, if they are significantly correlated, this means that there is conditional dependence!
	 */
	public static double[][] LEAD2 (Instances D, Result result) {

		int L = D.classIndex();
		int N = D.numInstances();
		double Y[][] = MLUtils.getYfromD(D);	//N*L					// Real
		//M.threshold(double[][],double)
		double Y_[][] = M.threshold(result.allPredictions(),0.5);	// Predicted
		// Error
		double E[][] = M.subtract(Y,Y_); 
		// Expected (for each j)
		double X[][] = new double[L][L];

		for(int j = 0; j < L; j++) {
			for(int k = j+1; k < L; k++) {
				for(int v : new int[]{0,1,-1}) { 
					//
					double p_j = p(E,j,v);								// prior
					double p_k = p(E,k,v);								// prior
					double p_jk = P(E,j,v,k,v);							// joint
					double Exp = p_j * p_k;									// expected
					//System.out.println("v = "+v);
					//System.out.println("p_j "+p_j);
					//System.out.println("p_k "+p_k);
					//System.out.println("p_jk"+p_jk);
					X[j][k] += ( ((p_jk - Exp) * (p_jk - Exp)) / Exp );		// calc.
				}
				//System.out.println(""+X[j][k]);
				X[j][k] -= CRITICAL[1];
			}
		}
		return X;
	}

	public static double[][] pearsonScore(double[][] L) {
		double M[][] = new double[L[0].length][L[0].length];
		for(int j = 0; j < L[0].length; j++) {
			double[] x = new double[L.length];

			for(int k = j + 1; k < L[0].length; k++) {
				double[] y = new double[L.length];
				for (int i = 0; i < L.length; i++) {
					x[i] = L[i][j];
					y[i] = L[i][k];
				}
				M[j][k] = pearsonScore(x, y);
			}
		}
		return M;
	}


	public static double pearsonScore(double[] x, double[] y) {
		int size = Math.min(x.length, y.length);
		double sumX = 0;
		double sumY = 0;
		for (int i = 0; i < size; i++) {
			sumX += x[i];
			sumY += y[i];
		}
		double xm = sumX / size;
		double ym = sumY / size;

		double nominator = 0;
		double d1 = 0;
		double d2 = 0;

		for (int i = 0; i < size; i++) {
			nominator += (x[i] - xm) * (y[i] - ym);
			d1 += Math.pow(x[i] - xm, 2);
			d2 += Math.pow(y[i] - ym, 2);
		}
		return Math.abs(nominator / (Math.sqrt(d1) * Math.sqrt(d2)));

	}



}
