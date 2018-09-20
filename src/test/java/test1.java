import dl4j.conf.MultiplyVertex;
import mulan.rbms.M;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class test1 {
    public static void main(String[] args) throws Exception {
        double lr = 0.001;
        int nnIn = 5;
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
//                .weightInit(WeightInit.RELU)
//                .activation(Activation.LEAKYRELU)
                .updater(new Adam(lr))
                .graphBuilder()
                .addInputs("input")
                .addLayer("dense1", new DenseLayer.Builder()
                        .nIn(nnIn)
                        .nOut(nnIn)
                        .activation(Activation.TANH)
                        .build(), "input")
                .addLayer("softmax", new ActivationLayer(Activation.SOFTMAX), "dense1")
                .addVertex("multiply", new MultiplyVertex(), "input", "softmax")
                .addLayer("output", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(nnIn)
                        .nOut(1)
                        .activation(Activation.SOFTMAX)
                        .build(), "multiply")
                .setOutputs("output")
                .build();
        ComputationGraph net = new ComputationGraph(config);
        net.init();
        System.out.println("Number of parameters by layer:");
        for(Layer l : net.getLayers() ){
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        double[][] netValues = new double[][]{
                {0.1, 100, 20, 3, 0.5, 1, 0},
                {0.7, 10, 91, 5, 0.1, 0, 1},
                {0.18, 102, 15, 2, 0.6, 1, 0},
                {0.4, 15, 89, 5, 0.3, 1, 1},
                {0.6, 100, 99, 6, 0.5, 1, 0},
                {0.67, 98, 22, 3, 0.5, 1, 0},
                {0.11, 13, 54, 3.9, 0.58, 0, 1},
                {0.15, 12, 99, 4.5, 0.65, 0, 1},
        };

        List<ListDataSetIterator<DataSet>> dp = new ArrayList<>();
        List<List<DataSet>> list = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            list.add(new ArrayList<DataSet>());
        }
        for (int i = 0; i < netValues.length; i++) {
            double[] features = Arrays.copyOfRange(netValues[i], 0 , 5);
            for (int j = 0; j < 2; j++) {
                double[] label = new double[]{netValues[i][5 + j]};
                DataSet d = new DataSet(Nd4j.create(features), Nd4j.create(label));
                list.get(j).add(d);
            }
        }
        for (int i = 0; i < list.size(); i++) {
            dp.add(new ListDataSetIterator<DataSet>(list.get(i)));
        }
        for (int i = 0; i < dp.size(); i++) {
            net.fit(dp.get(i), 100);
            Map<String, INDArray> paramTable = net.getLayer("dense1").paramTable();
            paramTable.forEach((k, v) -> {
                System.out.println(k + " : ");
                System.out.println(M.toString(v.toDoubleMatrix()));
            });

//            INDArray indArray = Nd4j.create(new double[]{0.1, 100, 20, 3, 0.5});
//            Map<String, INDArray> map = net.feedForward(indArray, false);
//            System.out.println(i + " :" + A.toString(map.get("multiply").data().asDouble()));
        }
    }
}
