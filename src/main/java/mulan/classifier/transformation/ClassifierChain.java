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
package mulan.classifier.transformation;

import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * <p>Implementation of the Classifier Chain (CC) algorithm.</p> <p>For more
 * information, see <em>Read, J.; Pfahringer, B.; Holmes, G.; Frank, E.
 * (2011) Classifier Chains for Multi-label Classification. Machine Learning.
 * 85(3):335-359.</em></p>
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @author Konstantinos Sechidis
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class ClassifierChain extends TransformationBasedMultiLabelLearner {

    /**
     * The new chain ordering of the label indices
     */
    private int[] chain;
    /**
     * The ensemble of binary relevance models. These are Weka
     * FilteredClassifier objects, where the filter corresponds to removing all
     * label apart from the one that serves as a target for the corresponding
     * model.
     */
    protected FilteredClassifier[] ensemble;

    /**
     * Creates a new instance using J48 as the underlying classifier
     */
    public ClassifierChain() {
        super(new J48());
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param aChain contains the order of the label indexes [0..numLabels-1] 
     */
    public ClassifierChain(Classifier classifier, int[] aChain) {
        super(classifier);
        chain = aChain;
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public ClassifierChain(Classifier classifier) {
        super(classifier);
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        ThreadPoolExecutor ec = new ThreadPoolExecutor(20, 20, 30, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
        if (chain == null) {
            chain = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                chain[i] = i;
            }
        }
        numLabels = train.getNumLabels();
        ensemble = new FilteredClassifier[numLabels];

        //把当前不是自分类器的标签列移除，只保留当前分类器对应的标签列
        for (int i = 0; i < numLabels; i++) {
            final int i1 = i;
            ec.submit(new Runnable() {
                @Override
                public void run() {
                    try {
                        MultiLabelInstances clone = train.clone();
                        Instances trainDataset = clone.getDataSet();
                        ensemble[i1] = new FilteredClassifier();
                        ensemble[i1].setClassifier(AbstractClassifier.makeCopy(baseClassifier));
                        // Indices of attributes to remove first removes numLabels attributes
                        // the numLabels - 1 attributes and so on.
                        // The loop starts from the last attribute.
                        int[] indicesToRemove = new int[numLabels - 1 - i1];
                        int counter2 = 0;
                        //将不是当前分类器的标签加入数组
                        for (int counter1 = 0; counter1 < numLabels - i1 - 1; counter1++) {
                            indicesToRemove[counter1] = labelIndices[chain[numLabels - 1 - counter2]];
                            counter2++;
                        }
                        //移除
                        Remove remove = new Remove();
                        remove.setAttributeIndicesArray(indicesToRemove);
                        remove.setInputFormat(trainDataset);
                        remove.setInvertSelection(false);
                        ensemble[i1].setFilter(remove);
                        //设置当前分类器对应的标签列作为标签列
                        trainDataset.setClassIndex(labelIndices[chain[i1]]);
                        ensemble[i1].buildClassifier(trainDataset);
                        debug("Bulding model " + (i1 + 1) + "/" + numLabels);
                        clone.getDataSet().delete();
                        clone = null;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }
        ec.shutdown();
        ec.awaitTermination(1, TimeUnit.DAYS);
        ec = null;
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];

        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        for (int counter = 0; counter < numLabels; counter++) {
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(tempInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
            bipartition[chain[counter]] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            // The confidence of the label being equal to 1
            confidences[chain[counter]] = distribution[classAttribute.indexOfValue("1")];

            tempInstance.setValue(labelIndices[chain[counter]], maxIndex);

        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
}