package dl4j;

import dl4j.conf.CopyVertex;
import dl4j.conf.MultiplyVertex;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.NormalizationFilter;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import mulan.util.Attention;
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
import org.deeplearning4j.nn.conf.layers.DenseLayer;
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
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.linalg.ops.transforms.Transforms;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.util.*;


import static org.deeplearning4j.nn.conf.Updater.RMSPROP;

public class TrainDl4j extends TransformationBasedMultiLabelLearner {

    /**
     * The number of classifier chain models
     */
    protected int numOfModels;
    /**
     * An array of ClassifierChain models
     */
    protected FilteredClassifier[] layer_1;
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
    /**
     * The size of each bag sample, as a percentage of the training size. Used
     * when useSamplingWithReplacement is true
     */
    protected int BagSizePercent = 100;
    private int featureLength;

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
    public TrainDl4j() {
        this(new J48(), new J48());
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     */
    public TrainDl4j(Classifier classifier, Classifier layer2) {
        super(classifier);
        layer2Clssifier= layer2;
        rand = new Random(1);
//        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

    }

    @Override
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        // dl4j
        Instances instances = train.getDataSet();
        numLabels = train.getNumLabels();
        numOfModels = numLabels;
        featureLength = featureIndices.length;
        int layer2FeatureLength = featureLength;


        List<ListDataSetIterator<DataSet>> dp = new ArrayList<>();

        List<List<DataSet>> list = new ArrayList<>();

        for (int i = 0; i < numLabels; i++) {
            list.add(new ArrayList<DataSet>());
        }

        int[] labelIndices = train.getLabelIndices();
        for (int i = 0; i < instances.numInstances(); i++) {
            System.out.println("当前处理：" + i +"/" + instances.numInstances() + "条记录.");
            double[] features = new double[featureLength];
            System.arraycopy(instances.get(i).toDoubleArray(), 0, features, 0 , featureLength);        ;
            for (int j = 0; j < numLabels; j++) {
                double label = instances.get(i).value(labelIndices[j]);
                double[] categoryLabel = label == 1.0 ? new double[]{0, 1} : new double[]{1, 0};
//                INDArray indArray = Nd4j.createSparseCOO(values, indexes, new long[]{instances.numAttributes()});
                INDArray indArray = Nd4j.create(features);
                DataSet d = new DataSet(indArray, Nd4j.create(categoryLabel));
                list.get(j).add(d.copy());
            }
        }


        for (int i = 0; i < numLabels; i++) {
            dp.add(new ListDataSetIterator<DataSet>(list.get(i), 1));
        }

        List<ComputationGraph> netList = new ArrayList<>();
        for (int i = 0; i < numLabels; i++) {
            ComputationGraph net = getComputationGraph(layer2FeatureLength);
            net.init();
            netList.add(net);
        }




        // fit dl4j for each label, and get attentions

        for (int i = 0; i < numLabels; i++) {
            List<EpochTerminationCondition> termimation = new ArrayList<>();
            termimation.add(new MaxEpochsTerminationCondition(3000));
            termimation.add(new ScoreImprovementEpochTerminationCondition(20, 0.002));
            EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder()
                    .epochTerminationConditions(termimation)
                    .scoreCalculator(new DataSetLossCalculator(dp.get(i), false))
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
    private ComputationGraph getComputationGraph(int dl4jInputLength) {
        int seed = 42;
        double lr = 0.01;
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
                        .activation(Activation.SOFTMAX)
                        .build(), "copy")
//                .addLayer("softmax", new ActivationLayer(Activation.SOFTMAX), "dense1")
                .addVertex("multiply", new MultiplyVertex(), "input", "dense1")
//                .addLayer("drop", new DropoutLayer.Builder(0.5).build(), "multiply")
                .addLayer("output", new OutputLayer.Builder()
                        .lossFunction(new LossNegativeLogLikelihood())
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
        INDArray attNd = Transforms.softmax(wxplusb);
        return x.mul(attNd).toDoubleVector();
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
        Instance metaInstance = null;

        //第一层 输出
        double[] layer_1_out = new double[numOfModels];
        for (int j = 0; j < numOfModels; j++) {
            double[] out = layer_1[j].distributionForInstance(instance);
            int predict = (out[0] > out[1]) ? 0 : 1;
            layer_1_out[j] = predict;
        }

        Instance normalizaIns = DataUtils.createInstance(train.getDataSet().instance(0), 1, instance.toDoubleArray());
        normalizationFilter.normalize(normalizaIns);
        double[] values = new double[featureLength + 2*numLabels];
        System.arraycopy(normalizaIns.toDoubleArray(), 0, values,0, featureLength);
        System.arraycopy(layer_1_out, 0, values, featureLength, numLabels);

        Instances layer_2_data = new Instances("layer_2", layer_2_Attr, 1);
//        double[] values = new double[featureLength + 2 * numLabels];
//        for (int m = 0; m < featureLength; m++) {
//            values[m] = instance.value(featureIndices[m]);
//        }
//        System.arraycopy(layer_1_out, 0, values, featureLength, numLabels);

        //将原标签向后移
        for (int j = 0; j < numLabels; j++) {
            values[labelIndices[j] + numLabels] = instance.value(labelIndices[j]);
        }
        metaInstance = DataUtils.createInstance(train.getDataSet().instance(0), 1, values);
        metaInstance.setDataset(layer_2_data);
        for (int i = 0; i < numOfModels; i++) {
            int index = chain.indexOf(i);
            double[] featureData = Arrays.copyOfRange(values, 0, featureLength + numLabels);
            INDArray input = Nd4j.create(featureData);
            INDArray[] output = neuralNetList[index].output(input);
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
            int maxIndex = (doubles[0] > doubles[1]) ? 0 : 1;
            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = layer_1[index].getFilter().getOutputFormat().classAttribute();

            bipartition[index] = classAttribute.value(maxIndex).equals("1");
            // The confidence of the label being equal to 1
            confidences[index] = doubles[classAttribute.indexOfValue("1")];
            // 更新数据
            values[labelIndices[index]] = maxIndex;

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
