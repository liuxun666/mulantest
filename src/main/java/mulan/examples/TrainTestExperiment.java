/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.examples;

import mulan.classifier.transformation.AttentionDoubleLayerCC;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Class demonstrating a simple train/test evaluation experiment
 *
 * @author Grigorios Tsoumakas
 * @version 2012.02.06
 */
public class TrainTestExperiment {

    /**
     * Executes this example
     *
     * @param args command-line arguments -path, -filestem and -percentage 
     * (training set), e.g. -path dataset/ -filestem emotions -percentage 67
     */
    public static void main(String[] args) {
        try {
            String path = "F:\\code\\mulan-master\\data\\multi-label\\birds\\";
            String filestem = "birds";
            String percentage = "80";
//
            System.out.println("Loading the dataset");
            MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + "-train.arff", path + filestem + ".xml");

            // split the data set into train and test
            Instances dataSet = mlDataSet.getDataSet();
            RemovePercentage rmvp = new RemovePercentage();
            rmvp.setInvertSelection(true);
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances trainDataSet = Filter.useFilter(dataSet, rmvp);

            rmvp = new RemovePercentage();
            rmvp.setPercentage(Double.parseDouble(percentage));
            rmvp.setInputFormat(dataSet);
            Instances testDataSet = Filter.useFilter(dataSet, rmvp);

            MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + filestem + ".xml");
            MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + filestem + ".xml");
//            MultiLabelInstances train = new MultiLabelInstances("F:\\code\\mulan-master\\data\\multi-label\\birds\\birds-train.arff", path + filestem + ".xml");
//            MultiLabelInstances test = new MultiLabelInstances("F:\\code\\mulan-master\\data\\multi-label\\birds" +
//                    "\\birds-test.arff", path + filestem + ".xml");

            Evaluator eval = new Evaluator();
            Evaluation results;
            Evaluation results1;

            Classifier brClassifier = new SMO();
//            BinaryRelevance br = new BinaryRelevance(brClassifier);
            AttentionDoubleLayerCC cc = new AttentionDoubleLayerCC(brClassifier, brClassifier);
//            br.setDebug(true);
            cc.setDebug(true);
            train.getDataSet();

            cc.getLabelRank(train);
//            br.build(train);
//            cc.build(train);
//            cc.makePrediction(train.getDataSet().instance(0));
//            results = eval.evaluate(br, test, train);
//            results1 = eval.evaluate(cc, test, train);
//            System.out.println(results);
//            System.out.println(results1);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}