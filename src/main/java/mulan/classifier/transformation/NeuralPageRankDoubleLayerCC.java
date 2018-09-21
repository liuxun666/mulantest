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

import dl4j.conf.MultiplyVertex;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.NormalizationFilter;
import mulan.data.MultiLabelInstances;
import mulan.rbms.M;
import mulan.util.A;
import mulan.util.Attention;
import mulan.util.StatUtils;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.jgrapht.alg.interfaces.VertexScoringAlgorithm;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DirectedWeightedPseudograph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.util.*;
import java.util.stream.Stream;

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
public class NeuralPageRankDoubleLayerCC extends TransformationBasedMultiLabelLearner {

    /**
     * The number of classifier chain models
     */
    protected int numOfModels;
    /**
     * An array of ClassifierChain models
     */
    protected FilteredClassifier[] layer_1;
    //第二层的链序
    protected ComputationGraph[] layer_2;

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
    private Classifier layer2Clssifier;


    private ArrayList<Attribute> layer_2_Attr;
    HashMap<Integer, Attention> attentions = new HashMap<>();


    /**
     * Default constructor
     */
    public NeuralPageRankDoubleLayerCC() {
        this(new J48(), new J48());
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     */
    public NeuralPageRankDoubleLayerCC(Classifier classifier, Classifier layer2) {
        super(classifier);
        layer2Clssifier= layer2;
        rand = new Random(1);
    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
//        IntStream stream = Arrays.stream(train.getLabelIndices());

        int[] cc = getCCChain(train);
        System.out.println(Arrays.toString(cc));
        for (int c: cc) {
            chain.add(c);
        }
        this.train = train;
        layer_2_Attr = layer_2_Attr();
        numOfModels = train.getNumLabels();
        layer_1 = new FilteredClassifier[numOfModels];
        layer_2 = new ComputationGraph[numLabels];

        Instances trainDataset;
        numLabels = train.getNumLabels();
        trainDataset = train.getDataSet();
        NormalizationFilter normalizationFilter = new NormalizationFilter(train, true, -1.0, 1.0);
        for (int i = 0; i < trainDataset.numInstances(); i++) {
            normalizationFilter.normalize(trainDataset.get(i));
        }

        //STEP1: 单独训练多个单分类器。
        //把当前不是自分类器的标签列移除，只保留当前分类器对应的标签列
        for (int i = 0; i < numLabels; i++) {
            layer_1[i] = new FilteredClassifier();
            layer_1[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));
            //移除
            int[] indicesToRemove = new int[numLabels - 1];
            int counter2 = 0;
            //将不是当前分类器的标签加入数组
            for (int counter1 = 0; counter1 < numLabels; counter1++) {
                if(counter1 != i){
                    indicesToRemove[counter2] = labelIndices[counter1];
                    counter2++;
                }
            }

            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainDataset);
            remove.setInvertSelection(false);
            layer_1[i].setFilter(remove);
            //设置当前分类器对应的标签列作为标签列
            trainDataset.setClassIndex(labelIndices[i]);
            debug("Bulding layer_1 model " + (i + 1) + "/" + numLabels);
            layer_1[i].buildClassifier(trainDataset);

        }
        //layer_1 done

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


        //build layer_2 data
        // data for BPMLL, input_size = train data featureindices + new feature(layer_1 output data) dim
        double[][] dataWithLayer1Output = new double[train.getNumInstances()][train.getDataSet().numAttributes() + numLabels];
        for (int i = 0; i < train.getNumInstances(); i++) {
//            double[] values = new double[train.getDataSet().numAttributes() + numLabels];
            for (int m = 0; m < featureIndices.length; m++) {
                dataWithLayer1Output[i][m] = train.getDataSet().instance(i).value(featureIndices[m]);
            }
            //将原标签向后移
            for (int j = 0; j < numLabels; j++) {
                dataWithLayer1Output[i][labelIndices[j] + numLabels] = train.getDataSet().instance(i).value(labelIndices[j]);
            }
            System.arraycopy(layer_1Predict[i], 0, dataWithLayer1Output[i], train.getDataSet().numAttributes(), numLabels);

        }



        // dl4j
        int dl4jInputLength = featureIndices.length + numLabels;


        List<ComputationGraph> netList = new ArrayList<>();
        for (int i = 0; i < numLabels; i++) {
            ComputationGraph net = getComputationGraph(dl4jInputLength);
            net.init();
            layer_2[i] = (net);
        }


        List<ListDataSetIterator<DataSet>> dp = new ArrayList<>();
        List<List<DataSet>> list = new ArrayList<>();
        for (int i = 0; i < numLabels; i++) {
            list.add(new ArrayList<DataSet>());
        }
        for (int i = 0; i < dataWithLayer1Output.length; i++) {
            double[] features = Arrays.copyOfRange(dataWithLayer1Output[i], 0 , dl4jInputLength);
            for (int j = 0; j < numLabels; j++) {
                double[] label = new double[]{dataWithLayer1Output[i][dl4jInputLength + j]};
                DataSet d = new DataSet(Nd4j.create(features), Nd4j.create(label));
                list.get(j).add(d);
            }
        }

        List<ListDataSetIterator<DataSet>> testDp = new ArrayList<>();
        int size = list.get(0).size();
        for (int i = 0; i < numLabels; i++) {
            List<DataSet> tmp = new ArrayList<>();
            for (int j = 0; j < (int)size * 0.2; j++) {
                tmp.add(list.get(i).remove(j));
            }
            testDp.add(new ListDataSetIterator<>(tmp));
            dp.add(new ListDataSetIterator<DataSet>(list.get(i)));
        }

        // fit dl4j for each label, and get attentions

        for (int i = 0; i < numLabels; i++) {
            EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(new MaxEpochsTerminationCondition(200))
                    .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(50, 0.0001))
                    .scoreCalculator(new DataSetLossCalculator(testDp.get(i), true))
                    .evaluateEveryNEpochs(1)
                    .modelSaver(new InMemoryModelSaver())
                    .build();
            EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, layer_2[i], dp.get(i));
            System.out.println("init attention for lable " + (i + 1));
            EarlyStoppingResult<ComputationGraph> fit = trainer.fit();
            System.out.println("EarlyStopping at " + fit.getTotalEpochs() + " epochs");
            System.out.println(fit.getTerminationDetails());
            ComputationGraph bestModel = fit.getBestModel();
            Evaluation evaluate = bestModel.evaluate(testDp.get(i));
            System.out.println("Evaluation " + i + " : " + evaluate);
            Map<String, INDArray> paramTable = bestModel.getLayer("dense1").paramTable();

//            netList.get(i).fit(dp.get(i), 200);
//            Map<String, INDArray> paramTable = net.getLayer("dense1").paramTable();
            INDArray w = paramTable.get("W");
            INDArray b = paramTable.get("b");
            attentions.put(i, new Attention(w, b));
        }


        System.out.println("dl4j end...");
        //建立第二层数据

//        for (int ii = 0; ii < numLabels; ii++) {
//            int index = chain.indexOf(ii);
//
//            Instances layer_2_data = new Instances("layer_2", layer_2_Attr, train.getNumInstances());
//            for (int i = 0; i < dataWithLayer1Output.length; i++) {
//                double[] fulldata = dataWithLayer1Output[i];
//
//                double[] featureData = Arrays.copyOfRange(fulldata, 0 , dl4jInputLength);
//                featureData = attentionData(featureData, index, dl4jInputLength);
//
//                System.arraycopy(featureData, 0, fulldata,0, featureData.length);
//                Instance ist = DataUtils.createInstance(train.getDataSet().instance(i), 1, fulldata);
//                ist.setDataset(layer_2_data);
//                layer_2_data.add(ist);
//
//            }
//
//
//
//            layer_2[index] = new FilteredClassifier();
//            layer_2[index].setClassifier(AbstractClassifier.makeCopy(layer2Clssifier));
//            int[] indicesToRemove = new int[numLabels - 1];
//            int counter2 = 0;
//            //将不是当前分类器的标签加入数组
//            for (int counter1 = 0; counter1 < numLabels; counter1++) {
//                if(counter1 != index){
//                    //新增了numLabels个特征在label之前， label的index需要加上numLabels
//                    indicesToRemove[counter2] = labelIndices[counter1] + numLabels;
//                    counter2++;
//                }
//            }
//
//            Remove remove = new Remove();
//            remove.setAttributeIndicesArray(indicesToRemove);
//            remove.setInputFormat(layer_2_data);
//            remove.setInvertSelection(false);
//            layer_2[index].setFilter(remove);
//            //设置当前分类器对应的标签列作为标签列
//            layer_2_data.setClassIndex(layer_2_data.numAttributes() - numLabels + index);
//            debug("Bulding layer_2 model " + (index + 1) + "/" + numLabels);
//            layer_2[index].buildClassifier(layer_2_data);
//            //更新数据
//            for (Instance ins : layer_2_data) {
//                double[] doubles = layer_2[index].distributionForInstance(ins);
//                int predict = (doubles[0] > doubles[1]) ? 0 : 1;
//                ins.setValue(layer_2_data.classIndex(), predict);
//            }
//        }

    }

    @NotNull
    private ComputationGraph getComputationGraph(int dl4jInputLength) {
        int seed = 42;
        double lr = 0.001;
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
//                .weightInit(WeightInit.RELU)
//                .activation(Activation.LEAKYRELU)
                .updater(new Adam(lr))
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(dl4jInputLength)
                        .nOut(dl4jInputLength)
                        .activation(Activation.SOFTMAX)
                        .build(), "input")
                .addVertex("multiply", new MultiplyVertex(), "input", "dense1")
                .addLayer("drop", new DropoutLayer.Builder(0.5).build(), "multiply")
                .addLayer("output", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(dl4jInputLength)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build(), "drop")
                .setOutputs("output")
                .build();
        return new ComputationGraph(config);
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

    private double[] attentionData(double[] featureData, int labelIndex, int featureLength) throws Exception {

        INDArray x = Nd4j.create(featureData);

        INDArray wxplusb = Transforms.tanh(x.mmuli(attentions.get(labelIndex).W).addiRowVector(attentions.get(labelIndex).b));
        INDArray attNd = Transforms.softmax(wxplusb);
        double[] data = x.mul(attNd).toDoubleVector();
        return data;
    }

    private int[] getCCChain(MultiLabelInstances train) {
        Stream<int[]> list = train.getDataSet().stream()
                .map(instance -> Arrays.stream(train.getLabelIndices()).map(i -> (int)instance.value(i)).toArray());
        int[][] labels = list.toArray(int[][]::new);

//        PageRank pageRank = new PageRank(labels).build();
//        double[] pr = pageRank.getPr().getColumnPackedCopy();
        double[] pr = pageRank(labels);
        System.out.println("pr" + Arrays.toString(pr));
        int root = maxIndex(pr);
//        int root = minIndex(pr);
//        int root = 0;
        double[][] ud = StatUtils.margDepMatrix(labels, train.getNumLabels());
//        double[][] ud = StatUtils.mInfomation(labels, train.getNumLabels());
        System.out.println("互信息矩阵 ：");
        System.out.println(M.toString(ud));
//        System.out.println(M.toString(ud1));
        EdgeWeightedGraph G = new EdgeWeightedGraph(train.getNumLabels());

        for(int i = 0; i < train.getNumLabels(); i++) {
            for(int j = i+1; j < train.getNumLabels(); j++) {
                Edge e = new Edge(i, j, -ud[i][j]);
                G.addEdge(e);
            }
        }

        KruskalMST mst = new KruskalMST(G);
        int paM[][] = new int[train.getNumLabels()][train.getNumLabels()];
        for (Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paM[j][k] = 1;
            paM[k][j] = 1;
            //StdOut.println(e);
        }

//        List<String> list = train.getLabelAttributes().stream().map(f -> f.value(0)).collect(Collectors.toList());
//        list.stream().limit(100).forEach(System.out::println);
//        new PageRank(trainDataset)

        System.out.println("Make a Tree from Root " + root);
        //这里paL[][]被初始化了，那么里面的元素就是全部0；
        int paL[][] = new int[train.getNumLabels()][0];
        int visted[] = new int[train.getNumLabels()];
        Arrays.fill(visted,-1);
        visted[root] = 0;
        //treeify(int:root=0树的根节点,int[标签数目][标签数目]：paM存储的是标签之间是否有边，有边则标签号交汇处值为1，int[标签数][0]：0，int[标签数]：visited记录标签是否被访问，[0，-1，-1，-1，-1，-1] )
        treeify(root,paM,paL, visted);
        return visted;
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

        double[] dataWithLayer1Output = new double[featureIndices.length + numLabels];
        System.arraycopy(instance.toDoubleArray(), 0 ,dataWithLayer1Output, 0 , featureIndices.length);
        System.arraycopy(layer_1_out, 0, dataWithLayer1Output, featureIndices.length, numLabels);
        for (int i = 0; i < numLabels; i++) {
            INDArray input = Nd4j.create(dataWithLayer1Output);
            INDArray output = layer_2[i].outputSingle(input);
            double[] doubles = output.toDoubleVector();
            int maxIndex = (doubles[0] > doubles[1]) ? 0 : 1;
            // Ensure correct predictions both for class values {0,1} and {1,0}

            bipartition[i] = instance.value(featureIndices.length  + i) == maxIndex;

            // The confidence of the label being equal to 1
            confidences[i] = output.getDouble(instance.value(featureIndices.length  + i) == 1 ? 0 : 1);
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

    public void treeify(int root, int paM[][], int paL[][], int visited[]) {
        int children[] = new int[]{};
        for(int j = 0; j < paM[root].length; j++) {
            if (paM[root][j] == 1) {
                if (visited[j] < 0) {
                    //A表示是一个方便数组操作的类，该处调用了该类的add方法
                    children = A.add(children,j);
                    paL[j] = A.add(paL[j],root);
                    //返回数组visited[]中最大元素的索引号
                    visited[j] = visited[maxIndex(visited)] + 1;
                }//endif
                // set as visited
                //paM[root][j] = 0;
            }//end1if
        }//endfor
        // go through again
        for(int child : children) {
            treeify(child,paM,paL,visited);
        }
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