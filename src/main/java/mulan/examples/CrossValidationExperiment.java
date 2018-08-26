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


import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Utils;

/**
 * Class demonstrating a simple cross-validation experiment
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.15
 */
public class CrossValidationExperiment {

    /**
     * Executes this example
     *
     * @param args command-line arguments -arff and -xml
     */
    public static void main(String[] args) {

        try {
            // e.g. -arff emotions.arff
            String arffFilename = "F:\\code\\mulan-master\\data\\multi-label\\emotions\\emotions.arff";
            // e.g. -xml emotions.xml
            String xmlFilename = "F:\\code\\mulan-master\\data\\multi-label\\emotions\\emotions.xml";

            System.out.println("Loading the dataset...");
            MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
            System.out.println(dataset.getNumInstances());
            Instance test = dataset.getDataSet().get(0);

            ClassifierChain br = new ClassifierChain(new NaiveBayes());
            br.build(dataset);

            MultiLabelOutput multiLabelOutput = br.makePrediction(test);
            System.out.println(multiLabelOutput);

//            RAkEL learner1 = new RAkEL(new LabelPowerset(new J48()));
//            MLkNN learner2 = new MLkNN();

//            Evaluator eval = new Evaluator();
//            MultipleEvaluation results;
//
//            int numFolds = 10;
//            results = eval.crossValidate(learner1, dataset, numFolds);
//            System.out.println(results);
//            results = eval.crossValidate(learner2, dataset, numFolds);
//            System.out.println(results);
        } catch (InvalidDataFormatException ex) {
            Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}