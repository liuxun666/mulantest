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

import mulan.classifier.transformation.*;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.PrintWriter;

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
            String path = "F:\\code\\mulan-master\\data\\multi-label\\medical\\";
            String filestem = "medical";
            String percentage = "80";
            System.out.println("Loading the dataset");
            MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

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

            System.out.println("train date set: " + trainDataSet.numInstances());
            System.out.println("test date set: " + testDataSet.numInstances());

            MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + filestem + ".xml");
            MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + filestem + ".xml");
//            MultiLabelInstances train = new MultiLabelInstances("F:\\code\\mulan-master\\data\\multi-label\\birds\\birds-train.arff", path + filestem + ".xml");
//            MultiLabelInstances test = new MultiLabelInstances("F:\\code\\mulan-master\\data\\multi-label\\birds" +
//                    "\\birds-test.arff", path + filestem + ".xml");

            Evaluator eval = new Evaluator();
            Evaluation result_br;
            Evaluation result_cc;
            Evaluation result_mbr;
            Evaluation result_dlmc;
            Evaluation result_atdcc;

            Classifier brClassifier = new SMO();
            BinaryRelevance br = new BinaryRelevance(brClassifier);
            ClassifierChain cc = new ClassifierChain(brClassifier);
            MultiLabelStacking mbr = new MultiLabelStacking(brClassifier, new SMO());
            AttMiPageRankNeuralNet atdcc = new AttMiPageRankNeuralNet(brClassifier, brClassifier);
            PageRankMiClassifierChain dlmc = new PageRankMiClassifierChain(brClassifier);
            dlmc.setDebug(true);
            atdcc.setDebug(true);
            br.build(train);
            mbr.build(train);
            cc.build(train);
            atdcc.build(train);
            dlmc.build(train);
            result_br = eval.evaluate(br, test, train);
            result_cc = eval.evaluate(cc, test, train);
            result_mbr = eval.evaluate(mbr, test, train);
            result_dlmc = eval.evaluate(dlmc, test, train);
            result_atdcc = eval.evaluate(atdcc, test, train);
            PrintWriter writer = new PrintWriter(new FileWriter("result/" + filestem + percentage));
            writer.println(filestem + ":");
            writer.println("br:");
            writer.println(result_br);
            writer.println("mbr:");
            writer.println(result_mbr);
            writer.println("cc:");
            writer.println(result_cc);
            writer.println("dlmc:");
            writer.println(result_dlmc);
            writer.println("atdcc:");
            writer.println(result_atdcc);
            writer.flush();
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}