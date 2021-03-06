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

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.BPMLL;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.util.ChainUtils;
import org.jgrapht.alg.interfaces.VertexScoringAlgorithm;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DirectedWeightedPseudograph;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * <p>Implementation of the Ensemble of Classifier Chains(ECC) algorithm.</p>
 * <p>For more information, see <em>Read, J.; Pfahringer, B.; Holmes, G., Frank,
 * E. (2011) Classifier Chains for Multi-label Classification. Machine Learning.
 * 85(3):335-359.</em></p>
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @author Konstantinos Sechidis
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class PageRankDoubleLayerCC extends TransformationBasedMultiLabelLearner {

    /**
     * The number of classifier chain models
     */
    protected int numOfModels;
    /**
     * An array of ClassifierChain models
     */
    protected FilteredClassifier[] layer_1;
    //第二层的链序
    protected FilteredClassifier[] layer_2;

    /**
     * Random number generator
     */
    protected Random rand;
    /**
     * Whether the output is computed based on the average votes or on the
     * average confidences
     */
    protected boolean useConfidences;
    /**
     * Whether to use sampling with replacement to create the data of the models
     * of the ensemble
     */
    protected boolean useSamplingWithReplacement = true;
    /**
     * The size of each bag sample, as a percentage of the training size. Used
     * when useSamplingWithReplacement is true
     */
    protected int BagSizePercent = 100;

    /**
     * Returns the size of each bag sample, as a percentage of the training size
     *
     * @return the size of each bag sample, as a percentage of the training size
     */
    public int getBagSizePercent() {
        return BagSizePercent;
    }

    /**
     * Sets the size of each bag sample, as a percentage of the training size
     *
     * @param bagSizePercent the size of each bag sample, as a percentage of the
     * training size
     */
    public void setBagSizePercent(int bagSizePercent) {
        BagSizePercent = bagSizePercent;
    }

    /**
     * Returns the sampling percentage
     *
     * @return the sampling percentage
     */
    public double getSamplingPercentage() {
        return samplingPercentage;
    }

    /**
     * Sets the sampling percentage
     *
     * @param samplingPercentage the sampling percentage
     */
    public void setSamplingPercentage(double samplingPercentage) {
        this.samplingPercentage = samplingPercentage;
    }
    /**
     * The size of each sample, as a percentage of the training size Used when
     * useSamplingWithReplacement is false
     */
    protected double samplingPercentage = 67;

    private MultiLabelInstances train;
    private ArrayList<Integer> chain = new ArrayList<>();


    private double[][][] layer_1Predict;

    private double[][] wights;
    private BPMLL bp;
    private Classifier layer2Clssifier;
    private ArrayList<Attribute> layer_2_Attr;
    private int featureLength;
    /**
     * Default constructor
     */
    public PageRankDoubleLayerCC() {
        this(new J48(), new J48());
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     */
    public PageRankDoubleLayerCC(Classifier classifier, Classifier layer2) {
        super(classifier);
        layer2Clssifier= layer2;
        rand = new Random(1);
    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        ThreadPoolExecutor ec = new ThreadPoolExecutor(20, 20, 30, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
        int[] list = ChainUtils.getMiCCChain(train);
        System.out.println(Arrays.toString(list));
        for (int c: list) {
            chain.add(c);
        }
        this.train = train;
        layer_2_Attr = layer_2_Attr();
        numOfModels = train.getNumLabels();
        layer_1 = new FilteredClassifier[numOfModels];
        layer_2 = new FilteredClassifier[numOfModels];
        featureLength = featureIndices.length;

        Instances trainDataset;
        numLabels = train.getNumLabels();
        trainDataset = train.getDataSet();

        //STEP1: 单独训练多个单分类器。
        //把当前不是自分类器的标签列移除，只保留当前分类器对应的标签列
        for (int i = 0; i < numLabels; i++) {
            int finalI = i;
            ec.submit(() -> {
                try {
                    Instances clone = train.clone().getDataSet();
                    layer_1[finalI] = new FilteredClassifier();
                    layer_1[finalI].setClassifier(AbstractClassifier.makeCopy(baseClassifier));
                    //移除
                    int[] indicesToRemove = new int[numLabels - 1];
                    int counter2 = 0;
                    //将不是当前分类器的标签加入数组
                    for (int counter1 = 0; counter1 < numLabels; counter1++) {
                        if(counter1 != finalI){
                            indicesToRemove[counter2] = labelIndices[counter1];
                            counter2++;
                        }
                    }

                    Remove remove = new Remove();
                    remove.setAttributeIndicesArray(indicesToRemove);
                    remove.setInputFormat(clone);
                    remove.setInvertSelection(false);
                    layer_1[finalI].setFilter(remove);
                    //设置当前分类器对应的标签列作为标签列
                    clone.setClassIndex(labelIndices[finalI]);
                    layer_1[finalI].buildClassifier(clone);
                    debug("Bulding layer_1 model " + (finalI + 1) + "/" + numLabels);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(1);
                }
            });


        }
        ec.shutdown();
        ec.awaitTermination(1, TimeUnit.DAYS);
        ec = null;

        //STEP2: 将多个单分类器分别输出对样本的预测
        double[][] layer_1Predict = new double[trainDataset.numInstances()][numLabels];
        for (int ii = 0; ii < trainDataset.numInstances(); ii++){
            double[] tmpScore = new double[numLabels];
            Instance nextElement = trainDataset.get(ii);
            for (int i = 0; i < numLabels; i++) {
                double[] doubles = layer_1[i].distributionForInstance(nextElement);
                tmpScore[i] = (doubles[0] > doubles[1]) ? 0 : 1;
            }
            layer_1Predict[ii] = tmpScore;
        }


//        //STEP3: 使用第一层的输出，输进神经网络，取中间的attention权重来给单分类器的输出加权。
//        List<DataPair> dp = new ArrayList<>();
//        for (int i = 0; i < layer_1Predict.length; i++) {
//            DataPair d = new DataPair(flatten(layer_1Predict[i]), getLabelValues(i));
//            dp.add(d);
//        }
//
//        bp = new BPMLL(42);
//        bp.setDebug(true);
//        bp.setTrainingEpochs(200);
//        bp.setHiddenLayers(new int[]{32, 16}); //test it
//        bp.build(dp, numLabels, train.getLabelIndices(), train.getLabelNames(), train.getFeatureIndices());
//        //建立第二层数据
//        double[][] layer_2_addData = new double[layer_1Predict.length][];
//        for (int i = 0; i < layer_1Predict.length; i++) {
//            //使用神经网络的输出并softmax  softmax(tanh(w*x +b))
//            layer_2_addData[i] = softmax(bp.predict(dp.get(i)).getConfidences());
//        }

        //layer_2 data
        Instances layer_2_data = new Instances("layer_2", layer_2_Attr, train.getNumInstances());
        for (int i = 0; i < train.getNumInstances(); i++) {
            double[] values = new double[layer_2_data.numAttributes()];
            //特征
            for (int m = 0; m < featureLength; m++) {
                values[m] = trainDataset.instance(i).value(featureIndices[m]);
            }
            //label 将原标签向后移
            for (int j = 0; j < numLabels; j++) {
                values[labelIndices[j] + numLabels] = trainDataset.instance(i).value(labelIndices[j]);
            }
            //使用第一层预测结果
            System.arraycopy(layer_1Predict[i], 0, values, featureLength, numLabels);

            Instance metaInstance = DataUtils.createInstance(trainDataset.instance(i), 1, values);
            metaInstance.setDataset(layer_2_data);

            layer_2_data.add(metaInstance);
        }



        for (int ii = 0; ii < numLabels; ii++) {

            int index = chain.indexOf(ii);

            layer_2[index] = new FilteredClassifier();
            layer_2[index].setClassifier(AbstractClassifier.makeCopy(baseClassifier));
            int[] indicesToRemove = new int[numLabels - 1];
            int counter2 = 0;
            //将不是当前分类器的标签加入数组
            for (int counter1 = 0; counter1 < numLabels; counter1++) {
                if(counter1 != index){
                    //新增了numLabels个特征在label之前， label的index需要加上numLabels
                    indicesToRemove[counter2] = labelIndices[counter1] + numLabels;
                    counter2++;
                }
            }

            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(layer_2_data);
            remove.setInvertSelection(false);
            layer_2[index].setFilter(remove);
            //设置当前分类器对应的标签列作为标签列
            layer_2_data.setClassIndex(labelIndices[index] + numLabels);
            debug("Bulding layer_2 model " + (index + 1) + "/" + numLabels);
            layer_2[index].buildClassifier(layer_2_data);
            //更新数据
            for (Instance ins : layer_2_data) {
                double[] doubles = layer_2[index].distributionForInstance(ins);
                int predict = (doubles[0] > doubles[1]) ? 0 : 1;
                ins.setValue(labelIndices[index], predict);
            }
        }

    }

    private double[] pageRank(int[][] labels) {
        double[][] PR1 = new double[labels[0].length][labels[0].length]; //出度
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[i].length; j++) {
                if(labels[i][j] == 1){
                    for (int l = 0; l < labels[i].length; l++) {
                        if(labels[i][l] == 1){
                            PR1[l][j] += 1;
                        }
                    }
                }
            }
        }

        DirectedWeightedPseudograph<String, DefaultEdge> g = new DirectedWeightedPseudograph<>(DefaultEdge.class);
        for (int i = 0; i < PR1.length; i++) {
            g.addVertex(String.valueOf(i));
        }
        for (int i = 0; i < PR1.length; i++) {
            for (int j = 0; j < PR1[i].length; j++) {
                if(PR1[i][j] > 0){
                    g.setEdgeWeight(g.addEdge(String.valueOf(i), String.valueOf(j)), PR1[i][j]);
                }
            }
        }
        VertexScoringAlgorithm<String, Double> pr = new org.jgrapht.alg.scoring.PageRank<>(g);
        double[] res = new double[PR1.length];
        for (int i = 0; i < res.length; i++) {
            res[i] = pr.getVertexScore(String.valueOf(i));
        }
        return res;

    }



    public double[] flatten(double[][] arr) {
        double[] tmp = new double[arr.length * arr[0].length];
        for (int i = 0; i < arr.length; i++) {
            System.arraycopy(arr[i], 0, tmp, i * arr[i].length, arr[i].length);
        }
        return tmp;
    }

    public double[] getLabelValues(int i) {
        int[] idxs = train.getLabelIndices();
        double[] tmp = new double[train.getLabelIndices().length];
        for (int j = 0; j < idxs.length; j++) {
            tmp[j] = train.getDataSet().get(i).value(idxs[j]);
        }

        return tmp;
    }

    public double[][] tanh(double[][] d) {
        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d[i].length; j++) {
                d[i][j] = StrictMath.tanh(d[i][j]);
            }
        }
        return d;
    }

    public static double[] softmax(double[] d) {
        double sum = 0;
        double[] tmp = new double[d.length];
        for (int i = 0; i < d.length; i++) {
            sum += StrictMath.pow(StrictMath.E, d[i]);
        }

        for (int i = 0; i < d.length; i++) {
            tmp[i] = StrictMath.pow(StrictMath.E, d[i]) / sum;
        }
        return tmp;
    }

    public ArrayList<Attribute> layer_2_Attr() {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();


        for (int j = 0; j < featureIndices.length; j++) {
            attributes.add(train.getDataSet().attribute(featureIndices[j]));
        }
        // add the labels in the last positions
        for (int j = 0; j < numLabels; j++) {
            attributes.add(train.getDataSet().attribute(labelIndices[j]));
        }

        for (int i = 0; i < numLabels; i++) {
            attributes.add(train.getDataSet().attribute(labelIndices[i]).copy("layer1_out_" + i));
        }
        return attributes;
    }


    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];
        Instance metaInstance = null;

        //第一层 输出
        double[] layer_1_out = new double[numOfModels];
        for (int j = 0; j < numOfModels; j++) {
            double[] out = layer_1[j].distributionForInstance(instance);
            int predict = (out[0] > out[1]) ? 0 : 1;
            layer_1_out[j] = predict;

        }

        Instances layer_2_data = new Instances("layer_2", layer_2_Attr, train.getNumInstances());
        double[] values = new double[featureLength + numLabels * 2];
        for (int m = 0; m < featureLength; m++) {
            values[m] = instance.value(featureIndices[m]);
        }
        System.arraycopy(layer_1_out, 0, values, featureLength, numLabels);
        metaInstance = DataUtils.createInstance(instance, 1, values);
        metaInstance.setDataset(layer_2_data);
        //将原标签向后移
        for (int j = 0; j < numLabels; j++) {
            values[labelIndices[j] + numLabels] = instance.value(labelIndices[j]);
        }

        for (int i = 0; i < numOfModels; i++) {

            int index = chain.indexOf(i);

            layer_2_data.setClassIndex(featureLength + numLabels + index);

            metaInstance.setClassValue(instance.value(labelIndices[index]));


            double[] doubles = layer_2[index].distributionForInstance(metaInstance);

            int maxIndex = (doubles[0] > doubles[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = layer_2[index].getFilter().getOutputFormat().classAttribute();
            bipartition[index] = classAttribute.value(maxIndex).equals("1");

            // The confidence of the label being equal to 1
            confidences[index] = doubles[classAttribute.indexOfValue("1")];

            //更新数据, 此时labelindex实际是第一层输出的index
            metaInstance.setValue(labelIndices[index], maxIndex);
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
//        MultiLabelOutput mlo = new MultiLabelOutput(predict, 0.5);
        return mlo;
    }

    public void getLabelRank(MultiLabelInstances mi) {
        int[] labelIndices = mi.getLabelIndices();
        double[] PR = new double[mi.getNumLabels()];

        Instances instances = mi.getDataSet();
        for (int i = 0; i < instances.numInstances(); i++) {
            double _sum = 0;
            for (int d: labelIndices) {
                _sum += instances.get(i).value(d);
            }
            for (int j = 0; j < mi.getNumLabels(); j++) {
                if(instances.get(i).value(labelIndices[j]) == 1.){
                    PR[j] += _sum - 1;
                }
            }
        }
        System.out.println(Arrays.toString(PR));
    }

    public double sum(double[] d){
        double sum = 0;
        for (double i: d) {
            sum += i;
        }
        return sum;
    }

    public int maxIndex(int[] a) {
        int init = a[0];
        int idx = 0;
        for (int i = 1; i < a.length; i++) {
            if(a[i] > init) {
                init = a[i];
                idx = i;
            }
        }
        return idx;
    }

    public int maxIndex(double[] a) {
        double init = a[0];
        int idx = 0;
        for (int i = 1; i < a.length; i++) {
            if(a[i] > init) {
                init = a[i];
                idx = i;
            }
        }
        return idx;
    }

    public int minIndex(double[] a) {
        double init = a[0];
        int idx = 0;
        for (int i = 1; i < a.length; i++) {
            if(a[i] < init) {
                init = a[i];
                idx = i;
            }
        }
        return idx;
    }


}