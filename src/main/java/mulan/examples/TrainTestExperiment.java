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

import mulan.classifier.transformation.AttMiPageRankNeuralNet;
import mulan.classifier.transformation.AttNeuralNet;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.PageRankDoubleLayerCC;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
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
//            String path = "data\\testData\\";
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
            Evaluation results2;
//            Evaluation results3;

            Classifier brClassifier = new SMO();
            AttMiPageRankNeuralNet cc = new AttMiPageRankNeuralNet(brClassifier, brClassifier);
            ClassifierChain br = new ClassifierChain(brClassifier);
            PageRankDoubleLayerCC pc = new PageRankDoubleLayerCC(brClassifier, brClassifier);
//            AttNeuralNet an = new AttNeuralNet(brClassifier, brClassifier);
            br.setDebug(true);
            cc.setDebug(true);
            cc.build(train);
            br.build(train);
            pc.build(train);
//            an.build(train);
//            results3 = eval.evaluate(an, test, train);
            results2 = eval.evaluate(cc, test, train);
            results = eval.evaluate(br, test, train);
            results1 = eval.evaluate(pc, test, train);
            System.out.println(results);
            System.out.println(results1);
            System.out.println(results2);
//            System.out.println(results3);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}