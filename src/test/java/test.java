import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import weka.core.matrix.Matrix;

import java.util.Arrays;

import static org.deeplearning4j.nn.conf.Updater.RMSPROP;

public class test {
    public static void main(String[] args) throws Exception {
        int lstmSize = 256;
        Bidirectional encode = new Bidirectional.Builder(null, null)
                .mode(Bidirectional.Mode.CONCAT)
                .rnnLayer(new LSTM.Builder()
                        .nIn(lstmSize)
                        .nOut(lstmSize)
                        .build()
                ).build();
        Bidirectional decode = new Bidirectional.Builder(null, null)
                .mode(Bidirectional.Mode.CONCAT)
                .rnnLayer(new LSTM.Builder()
                        .nIn(lstmSize)
                        .nOut(lstmSize)
                        .build()
                ).build();
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
//                .weightInit(WeightInit.RELU)
//                .activation(Activation.LEAKYRELU)
                .updater(RMSPROP)
                .graphBuilder()
                .addInputs("input")
                .addLayer("encode", encode, "input"
                )
                .addLayer("lastTime", new LastTimeStep(decode), "decode")
                .addLayer("wh", new DenseLayer.Builder().nIn(2 * lstmSize).nOut(2*lstmSize).build(), "encode")
                .addLayer("ws", new DenseLayer.Builder().nIn(2 * lstmSize).nOut(2*lstmSize).build(), "lastTime")
//                .addLayer("concat", new )
                .addLayer("e", new ActivationLayer(Activation.TANH), "wh")
                .addLayer("decode", decode, "input"
                )
                .setOutputs("output")
                .build();
        new ComputationGraph(config);


    }

    public static boolean compareAbs(Matrix a, Matrix b){

        boolean flag = true;
        for(int i = 0;i < a.getRowDimension();i++){
            for(int j = 0;j < a.getColumnDimension();j++){
                if(Math.abs(a.get(i, j)) <= Math.abs(b.get(i, j))){
                    flag = false;break;
                }
            }
        }
        return flag;
    }


    public static double sum(double[] d){
        double sum = 0;
        for (double i: d) {
            sum += i;
        }
        return sum;
    }

    public static  double[][] tanh(double[][] d) {
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

    private static class PageRank {
        private double[][] data;
        private Matrix pr;
        private Matrix pageRankPre;
        private int iterator;

        public PageRank(double[][] data) {
            this.data = data;
        }

        public Matrix getPr() {
            return pr;
        }


        public int getIterator() {
            return iterator;
        }

        public PageRank build() {
            double[][] PR1 = new double[data[0].length][data[0].length]; //出度
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    if(data[i][j] == 1){
                        for (int l = 0; l < data[i].length; l++) {
                            if(data[i][l] == 1){
                                PR1[l][j] += 1;
                            }
                            if(j == l){
                                PR1[j][l] = 0;
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < PR1.length; i++) {
                System.out.println(Arrays.toString(PR1[i]));
            }

            System.out.println();

            for (int i = 0; i < PR1.length; i++) {
                double sum = 0d;
                for (int j = 0; j < PR1.length; j++) {
                    sum += PR1[j][i];
                }
                for (int j = 0; j < PR1.length; j++) {
                    PR1[j][i] = PR1[j][i] / sum;
                }
            }

            for (int i = 0; i < PR1.length; i++) {
                System.out.println(Arrays.toString(PR1[i]));
            }

            int n = PR1.length;
            double alpha = 0.85;
            Matrix ma = new Matrix(PR1);//源矩阵
            pr = new Matrix(n, 1, 1.0d);
            Matrix minimum = new Matrix(n ,1, 0.000001d);//极小值
            Matrix u = new Matrix(n, n,1.0d);//单元矩阵


            pageRankPre = pr;
            Matrix g = ma.times(alpha).plus(u.times((1-alpha)/n));
            pr = g.times(pr);
            iterator = 1;
            while(true){
                if(compareAbs(minimum, pageRankPre.minus(pr))){
                    break;
                }else{
                    pageRankPre = pr;
                    g = ma.times(alpha).plus(u.times((1-alpha)/n));
                    pr = g.times(pr);
                    iterator ++;
                }
            }
            pageRankPre = null;
            return this;
        }
    }


//    public void test() {
//        double[] layerInput = new double[]{0.12, 0.12, 0.15, 0.13, 0.18, 0.14};
//        List<Neuron> layer = layers.get(layerIndex);
//        int layerSize = layer.size();
//        layerOutput = new double[layerSize];
//        for (int n = 0; n < layerSize; n++) {
//            if (layerIndex == 0) {
//                layerOutput[n] = layer.get(n).processInput(new double[]{layerInput[n]});
//            } else {
//                layerOutput[n] = layer.get(n).processInput(layerInput);
//            }
//        }
//        layerInput = Arrays.copyOf(layerOutput, layerOutput.length);
//    }
}
