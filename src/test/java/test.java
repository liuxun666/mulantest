import mulan.classifier.neural.BPMLL;
import mulan.classifier.neural.DataPair;
import mulan.classifier.neural.model.Neuron;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelNodeImpl;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import mulan.data.generation.NumericAttribute;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class test {
    public static void main(String[] args) throws Exception {
        double[][] data = new double[5][4];
        data[0] = new double[]{1., 0., 0., 1.};
        data[1] = new double[]{1., 1., 0., 1.};
        data[2] = new double[]{0., 0., 1., 1.};
        data[3] = new double[]{1., 0., 1., 0.};
        data[4] = new double[]{1., 0., 1., 0.};
        double[] PR = new double[4];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                if(data[i][j] == 1.){
                    PR[j] = PR[j] + sum(data[i]) - 1;
                }
            }
        }
        System.out.println(Arrays.toString(PR));
//        double[][] wights = bp.getWights();
//        Matrix w = new Matrix(wights);
//        System.out.println(w); // 4*7
//
//        double[][] da = new double[3][];
//        da[0] = new double[]{0.6, 0, 0.65};
//        da[1] = new double[]{0.8, 0.7, 0.4};
//        da[2] = new double[]{0.9, 0.3, 0.8};
//        Matrix m = new Matrix(da); //3*3
//        System.out.println(new Matrix(tanh(m.times(w).getArray()))); //3 * 7
//        Matrix tanh = new Matrix(tanh(m.times(w).getArray()));




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
