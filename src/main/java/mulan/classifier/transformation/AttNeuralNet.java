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

import dl4j.conf.CopyVertex;
import dl4j.conf.MultiplyVertex;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.NormalizationFilter;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.rbms.M;
import mulan.util.A;
import mulan.util.Attention;
import mulan.util.StatUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.jetbrains.annotations.NotNull;
import org.jgrapht.alg.interfaces.VertexScoringAlgorithm;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DirectedWeightedPseudograph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.linalg.ops.transforms.Transforms;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Stream;

import static org.deeplearning4j.nn.conf.Updater.RMSPROP;

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
public class AttNeuralNet extends TransformationBasedMultiLabelLearner {

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
    protected ComputationGraph[] neuralNetList;
    protected NormalizationFilter normalizationFilter;

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
    private int featureLength;

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
    public AttNeuralNet() {
        this(new J48(), new J48());
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     */
    public AttNeuralNet(Classifier classifier, Classifier layer2) {
        super(classifier);
        layer2Clssifier= layer2;
        rand = new Random(1);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
//        IntStream stream = Arrays.stream(train.getLabelIndices());

        this.train = train;
        numLabels = train.getNumLabels();
        numOfModels = numLabels;
        featureLength = featureIndices.length;
        neuralNetList = new ComputationGraph[numLabels];

        Instances trainDataset;
        trainDataset = train.getDataSet();
        MultiLabelInstances normalizaTrain = train.clone();
        normalizationFilter = new NormalizationFilter(normalizaTrain, true, 0.0, 1.0);

        //build layer_2 data
        // data for BPMLL, input_size = train data numAttributes + new feature(layer_1 output data) dim
        double[][] dataForNeural = new double[train.getNumInstances()][trainDataset.numAttributes() + numLabels];
        double[][] class_weight = new double[numLabels][2];
        for (int i = 0; i < train.getNumInstances(); i++) {
//            double[] values = new double[train.getDataSet().numAttributes() + numLabels];
            System.arraycopy(normalizaTrain.getDataSet().get(i).toDoubleArray(), 0, dataForNeural[i], 0, featureLength + numLabels);
            for (int j = 0; j < numLabels; j++) {
                double labelValue = trainDataset.get(i).value(labelIndices[j]);
                if (labelValue > 0) {
                    class_weight[j][1] += 1;
                }else{
                    class_weight[j][0] += 1;
                }
            }

        }
        for (int i = 0; i < class_weight.length; i++) {
            double max = class_weight[i][0] > class_weight[i][1] ? class_weight[i][0] : class_weight[i][1];

            class_weight[i][0] = max / class_weight[i][0] == 0 ? max : class_weight[i][0];
            class_weight[i][1] = max / class_weight[i][1] == 0 ? max : class_weight[i][1];
        }
        System.out.println(M.toString(class_weight));

        //class weight for dl4j


        // dl4j
        List<ComputationGraph> netList = new ArrayList<>();
        for (int i = 0; i < numLabels; i++) {
            ComputationGraph net = getComputationGraph(featureLength, Nd4j.create(class_weight[i]));
            net.init();
//            net.addListeners(new ScoreIterationListener(1000));
            //Initialize the user interface backend
            UIServer uiServer = UIServer.getInstance();
            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
            //Then add the StatsListener to collect this information from the network, as it trains
            net.setListeners(new StatsListener(statsStorage));
            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage);
            netList.add(net);
        }


        List<ListDataSetIterator<DataSet>> dp = new ArrayList<>();
        List<List<DataSet>> list = new ArrayList<>();
        for (int i = 0; i < numLabels; i++) {
            list.add(new ArrayList<DataSet>());
        }
        for (int i = 0; i < dataForNeural.length; i++) {
            double[] features = Arrays.copyOfRange(dataForNeural[i], 0 , featureLength);
            for (int j = 0; j < numLabels; j++) {
                double label = dataForNeural[i][featureLength + j];
                double[] categoryLabel = label == 1.0 ? new double[]{0, 1} : new double[]{1, 0};
                DataSet d = new DataSet(Nd4j.create(features), Nd4j.create(categoryLabel));
                list.get(j).add(d.copy());
            }
        }


        for (int i = 0; i < numLabels; i++) {
            dp.add(new ListDataSetIterator<DataSet>(list.get(i), 32));
        }

        // fit dl4j for each label, and get attentions

        for (int i = 0; i < numLabels; i++) {
            List<EpochTerminationCondition> termimation = new ArrayList<>();
            termimation.add(new MaxEpochsTerminationCondition(2000));
            termimation.add(new ScoreImprovementEpochTerminationCondition(100, 0.001));
            EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(termimation)
                    .scoreCalculator(new DataSetLossCalculator(dp.get(i), true))
                    .evaluateEveryNEpochs(10)
                    .modelSaver(new InMemoryModelSaver())
                    .build();
            EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, netList.get(i), dp.get(i));
            System.out.println("init attention for lable " + (i + 1));

//            netList.get(i).fit(dp.get(i), 4000);
            EarlyStoppingResult<ComputationGraph> fit = trainer.fit();
            System.out.println("EarlyStopping at " + fit.getTotalEpochs() + " epochs");
            ComputationGraph bestModel = fit.getBestModel();
            neuralNetList[i] = bestModel;
            Evaluation evaluate = bestModel.evaluate(dp.get(i));
            System.out.println("Evaluation " + i + " : " + evaluate);
//            Map<String, INDArray> paramTable = netList.get(i).getLayer("dense1").paramTable();
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
//            Instances layer_2_data = new Instances("layer_2", layer_2_Attr, train.getNumInstances());
//            for (int i = 0; i < dataWithLayer1Output.length; i++) {
//                double[] fulldata = dataWithLayer1Output[i];
//
//                double[] featureData = Arrays.copyOfRange(fulldata, 0 , layer2FeatureLength);
//                featureData = attentionData(featureData, index);
//
//                System.arraycopy(featureData, 0, fulldata,0, featureData.length);
////                Instance ist = DataUtils.createInstance(trainDataset.firstInstance(), 1, fulldata);
//                Instance ist = DataUtils.createInstance(normalizaTrain.getDataSet().firstInstance(), 1, fulldata);
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
//            layer_2_data.setClassIndex(layer2FeatureLength + index);
//            debug("Bulding layer_2 model " + (index + 1) + "/" + numLabels);
//            layer_2[index].buildClassifier(layer_2_data);
//            //更新数据
//            for (int i = 0; i < dataWithLayer1Output.length; i++) {
//                double[] doubles = layer_2[index].distributionForInstance(layer_2_data.instance(i));
//                int predict = (doubles[0] > doubles[1]) ? 0 : 1;
//                dataWithLayer1Output[i][featureLength + index] = (double)predict;
//            }
//        }

    }

    @NotNull
    private ComputationGraph getComputationGraph(int dl4jInputLength, INDArray class_weight) {
        int seed = 42;
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
//                .weightInit(WeightInit.RELU)
//                .activation(Activation.LEAKYRELU)
                .updater(RMSPROP)
                .graphBuilder()
                .addInputs("input")
                .addVertex("copy", new CopyVertex(), "input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(dl4jInputLength)
                        .nOut(dl4jInputLength)
                        .activation(Activation.TANH)
                        .build(), "copy")
                .addLayer("softmax", new ActivationLayer(Activation.SOFTMAX), "dense1")
                .addVertex("multiply", new MultiplyVertex(), "input", "softmax")
//                .addLayer("drop", new DropoutLayer.Builder(0.5).build(), "multiply")
                .addLayer("output", new OutputLayer.Builder()
                        .lossFunction(new LossNegativeLogLikelihood(class_weight))
                        .nIn(dl4jInputLength)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build(), "multiply")
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

    private double[] attentionData(double[] featureData, int labelIndex) throws Exception {
        INDArray x = Nd4j.create(featureData);
        INDArray wxplusb = x.mmul(attentions.get(labelIndex).W).addRowVector(attentions.get(labelIndex).b);
        INDArray attNd = Transforms.softmax(Transforms.tanh(wxplusb));
        return x.mul(attNd).toDoubleVector();
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


        for (int j = 0; j < train.getFeatureIndices().length; j++) {
            attributes.add(train.getDataSet().attribute(featureIndices[j]));
        }

        for (int i = 0; i < numLabels; i++) {
            attributes.add(train.getDataSet().attribute(labelIndices[i]).copy("layer1_out_" + i));
        }
        // add the labels in the last positions
        for (int j = 0; j < numLabels; j++) {
            attributes.add(train.getDataSet().attribute(labelIndices[j]));
        }

        return attributes;
    }


    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception{

        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];

        Instance normalizaIns = DataUtils.createInstance(train.getDataSet().instance(0), 1, instance.toDoubleArray());
        normalizationFilter.normalize(normalizaIns);
        double[] values = normalizaIns.toDoubleArray();

        for (int i = 0; i < numOfModels; i++) {
            double[] featureData = Arrays.copyOfRange(values, 0, featureLength);
            INDArray input = Nd4j.create(featureData);
            INDArray[] output = neuralNetList[i].output(input);
//            double[] featureData = Arrays.copyOfRange(values, 0 , featureLength + numLabels);
//            System.out.println("before Attention: " + A.toString(featureData));
//            featureData = attentionData(featureData, index);
//            System.out.println("after Attention: " + A.toString(featureData));
//            System.out.println("weigth: " + M.toString(attentions.get(i).W.toDoubleMatrix()));
//            System.out.println("bias: " + A.toString(attentions.get(i).b.toDoubleVector()));
//            for (int j = 0; j < featureData.length; j++) {
//                metaInstance.setValue(j, featureData[j]);
//            }
//            layer_2_data.setClassIndex(featureLength + numLabels + index);
//            metaInstance.setClassValue(instance.value(labelIndices[index]));
            double[] doubles = output[0].toDoubleVector();
            int yPredict = (doubles[0] > doubles[1]) ? 0 : 1;
            // Ensure correct predictions both for class values {0,1} and {1,0}
//            Attribute classAttribute = layer_1[index].getFilter().getOutputFormat().classAttribute();
            int yTrue = (int)values[featureLength + i];
//            bipartition[chain[counter]] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            bipartition[i] = yTrue == yPredict;
            // The confidence of the label being equal to 1
//            confidences[index] = doubles[classAttribute.indexOfValue("1")];
            confidences[i] = doubles[1];
            // 更新数据

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