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

import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;

public abstract class MLEvalUtils {

	/**
	 * GetThreshold.
	 * Get a threshold from a Threshold OPtion string 'top'.
	 * @param	Y	label space; for calculating a threshold with PCut
	 * @param	D	training data; for calculating a threshold with PCut
	 * @param	top Threshold OPtion (either "PCut1", "PCutL" or a real value e.g. "0.5" or L real values e.g. "[0.1, 0.2, 0.8]" for L = 3
	 */
	public static String getThreshold(ArrayList<double[]> Y, Instances D, String top) throws Exception {
		if (top.equals("PCut1") || top.equals("c")) {			// Proportional Cut threshold (1 general threshold)
			return String.valueOf(ThresholdUtils.calibrateThreshold(Y,MLUtils.labelCardinality(D)));
		}	
		else if (top.equals("PCutL") || top.equals("C")) {		// Proportional Cut thresholds (one for each Label)
			return Arrays.toString(ThresholdUtils.calibrateThresholds(Y,MLUtils.labelCardinalities(D)));
		}
		else { 								// Set our own threshold (we assume top = "0.5" or top = "[0.1,...,0.3]"
											// (we make no checks here!) // X return String.valueOf(Double.parseDouble(top));
			return top;
		}
	}

	/**
	 * GetMLStats.
	 * Given predictions and corresponding true values and a threshold string, retreive statistics.
	 * @param	Confidences	predictions (may be real-valued confidences)
	 * @param	TrueValues	corresponding true values
	 * @param	t			a threshold string, e.g. "0.387"
	 * @return	the evaluation statistics
	 */
	//public static HashMap<String,Double> getMLStats(ArrayList<double[]> Confidences, ArrayList<int[]> TrueValues, String t) {
	public static HashMap<String,Double> getMLStats(double Rpred[][], int Y[][], String t, String vop) {
		double ts[] = ThresholdUtils.thresholdStringToArray(t,Y[0].length);
		/*
		if (t.startsWith("[")) {
			ts = MLUtils.toDoubleArray(t);							// threshold vector       [t1 t2 ... tL]]
		}
		else {
			ts = new double[Y[0].length];
			Arrays.fill(ts,Double.parseDouble(t));					// make a threshold vector [t t t ... t]
		}
		*/
		return getMLStats(Rpred,Y,ts,vop);
	}

	/**
	 * GetMLStats.
	 * Given predictions and corresponding true values and a threshold string, retreive statistics.
	 * @param	Confidences	predictions (may be double-valued confidences in the multi-label case)
	 * @param	TrueValues	corresponding true values
	 * @param	t			a vector of thresholds, e.g. [0.1,0.1,0.1] or [0.1,0.5,0.4,0.001]
	 * @return	the evaluation statistics
	 */
	public static HashMap<String,Double> getMLStats(double Rpred[][], int Y[][], double t[], String vop) {

		int N = Y.length; 
		int L = Y[0].length;

		int V = MLUtils.getIntegerOption(vop,1); // default 1

		int Ypred[][] = ThresholdUtils.threshold(Rpred,t);

		HashMap<String,Double> results = new LinkedHashMap<String,Double>();

		//results.put("N_test"			,(double)N);
		results.put("L"					,(double)L);
		results.put("Accuracy"			,Metrics.P_Accuracy(Y,Ypred));
		results.put("Hamming score"		,Metrics.P_Hamming(Y,Ypred));
		results.put("Exact match"		,Metrics.P_ExactMatch(Y,Ypred));

		if (V > 1) {

			results.put("Hamming loss"		,Metrics.L_Hamming(Y,Ypred));
			results.put("ZeroOne loss"		,Metrics.L_ZeroOne(Y,Ypred));
			results.put("One error"			,Metrics.L_OneError(Y,Rpred));
			results.put("Rank loss"			,Metrics.L_RankLoss(Y,Rpred));
			results.put("Avg precision"		,Metrics.P_AveragePrecision(Y,Rpred));
			results.put("Log Loss D"		,Metrics.L_LogLossD(Y,Rpred));
			results.put("Log Loss L"		,Metrics.L_LogLossL(Y,Rpred));
			//results.put("Precision"		,Metrics.P_Precision(Y,Ypred));
			//results.put("Recall"			,Metrics.P_Recall(Y,Ypred));
			results.put("F-micro"			,Metrics.P_FmicroAvg(Y,Ypred));
			results.put("F-macro_D"			,Metrics.P_FmacroAvgD(Y,Ypred));
			results.put("F-macro_L"			,Metrics.P_FmacroAvgL(Y,Ypred));
			results.put("N_empty"			,MLUtils.emptyVectors(Ypred));
			results.put("LCard_pred"		,MLUtils.labelCardinality(Ypred));

			if (V > 2) {
				//results.put("LCard_real"		,MLUtils.labelCardinality(Y));
				for(int j = 0; j < L; j++) {
					results.put("Accuracy["+j+"]"	,1.0-Metrics.P_Hamming(Y,Ypred,j));
				}
			}
			if (V > 3) {
				// @todo: show by-label label cardinality
			}
		}
		return results;
	}

	/**
	 * GetMTStats.
	 * Given multi-target predictions and corresponding true values, retreive evaluation statistics.
	 * @param	Predictions	predictions
	 * @param	TrueValues	corresponding true values
	 * @return	the evaluation statistics
	 */
	public static HashMap<String,Double> getMTStats(double Rpred[][], int Y[][], String vop) {

		// just a question of rounding for now
		int Ypred[][] = ThresholdUtils.round(Rpred);

		int N = Y.length;
		int L = Y[0].length;
		int V = MLUtils.getIntegerOption(vop,1); // default 1

		HashMap<String,Double> output = new LinkedHashMap<String,Double>();
		output.put("N"					,(double)N);
		output.put("L"					,(double)L);
		output.put("Hamming loss"		,Metrics.L_Hamming(Y,Ypred));
		output.put("ZeroOne loss"		,Metrics.L_ZeroOne(Y,Ypred));
		output.put("Hamming score"		,Metrics.P_Hamming(Y,Ypred));
		output.put("Exact match"		,Metrics.P_ExactMatch(Y,Ypred));

		if (V > 2) {
			for(int j = 0; j < L; j++) {
				output.put("L"+j+"_acc"				,1.0 - Metrics.P_Hamming(Y,Ypred,j));
			}
		}
		return output;
	}

	/**
	 * AverageResults - Create a Result with the average of an array of Results by taking the average +/- standand deviation.
	 * @param	folds	array of Results (e.g., from CV-validation)
	 * @return	A result reporting the average of these folds.
	 */
	public static Result averageResults(Result folds[]) { 
		Result r = new Result();
		// for info ..
		r.info = folds[0].info;
		// for output ..
		for(String metric : folds[0].output.keySet()) {
			double values[] = new double[folds.length];
			for(int i = 0; i < folds.length; i++) {
				values[i] = folds[i].output.get(metric);
			}
			String avg_sd = Utils.doubleToString(Utils.mean(values),5,3)+" +/- "+Utils.doubleToString(Math.sqrt(Utils.variance(values)),5,3);
			r.setInfo(metric,avg_sd);
		}
		// and now for 'vals' ..
		for(String metric : folds[0].vals.keySet()) {
			double values[] = new double[folds.length];
			for(int i = 0; i < folds.length; i++) {
				values[i] = folds[i].vals.get(metric);
			}
			String avg_sd = Utils.doubleToString(Utils.mean(values),5,3)+" +/- "+Utils.doubleToString(Math.sqrt(Utils.variance(values)),5,3);
			r.setInfo(metric,avg_sd);
		}
		r.setInfo("Type","CV");
		return r;
	}

	public static void main(String args[]) {
	}

}
