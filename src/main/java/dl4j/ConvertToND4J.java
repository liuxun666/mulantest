package dl4j;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import org.datavec.api.transform.TransformProcess;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class ConvertToND4J {

    public static void main(String[] args) throws Exception {
        MultiLabelInstances train = null;
        MultiLabelInstances test = null;
        String path = args[0];
        String fileName = args[1];
        String percentage = "80";
        System.out.println("Loading the dataset " + fileName);
        MultiLabelInstances mlDataSet;
        mlDataSet = new MultiLabelInstances(path + "/" + fileName + ".arff", path + "/" + fileName + ".xml");

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

        train = new MultiLabelInstances(trainDataSet, path + "/" + fileName + ".xml");
        test = new MultiLabelInstances(testDataSet, path + "/" + fileName + ".xml");

        int numLabels = train.getNumLabels();

        List<ListDataSetIterator<DataSet>> dp = new ArrayList<>();



        convert2File(train, numLabels, "train");
        convert2File(train, numLabels, "test");


    }

    private static void convert2File(MultiLabelInstances train, int numLabels, String shufix) throws IOException {
        List<List<DataSet>> list = new ArrayList<>();
        for (int i = 0; i < numLabels; i++) {
            list.add(new ArrayList<DataSet>());
        }
        Instances instances = train.getDataSet();
        int featureLength = train.getFeatureIndices().length;
        int[] labelIndices = train.getLabelIndices();
        AtomicInteger count = new AtomicInteger();
        instances.stream().parallel().forEach(ins -> {
            double[] features = new double[featureLength];
            System.arraycopy(ins.toDoubleArray(), 0, features, 0 , featureLength);        ;
            for (int j = 0; j < numLabels; j++) {
                double label = ins.value(labelIndices[j]);
                double[] categoryLabel = label == 1.0 ? new double[]{0, 1} : new double[]{1, 0};
//                INDArray indArray = Nd4j.createSparseCOO(values, indexes, new long[]{instances.numAttributes()});
                INDArray indArray = Nd4j.create(features);
                DataSet d = new DataSet(indArray, Nd4j.create(categoryLabel));
                list.get(j).add(d.copy());
            }
            int c = count.incrementAndGet();
            if(c % 1000 == 0){
                System.out.println("完成第" + c + "个");
            }
        });

        for (int i = 0; i < numLabels; i++) {
            System.out.println("保存第" + i + "个文件");
            ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream("data/dataset_" + shufix + "_0000" +i + ".data"));
            ListDataSetIterator<DataSet> dataSetIterator = new ListDataSetIterator<>(list.get(i), 128);
            outputStream.writeObject(dataSetIterator);
            outputStream.flush();
            outputStream.close();
        }
    }

    static void getInstans(String path, String fileName, int numLabels, MultiLabelInstances train, MultiLabelInstances test) throws Exception {
        String percentage = "80";
        System.out.println("Loading the dataset " + fileName);
        MultiLabelInstances mlDataSet;
        if(numLabels == 0){
            mlDataSet = new MultiLabelInstances(path + "/" + fileName + ".arff", path + "/" + fileName + ".xml");
        }else{
            mlDataSet = new MultiLabelInstances(path + "/" + fileName + ".arff", numLabels);
        }

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

        train = new MultiLabelInstances(trainDataSet, path + "/" + fileName + ".xml");
        test = new MultiLabelInstances(testDataSet, path + "/" + fileName + ".xml");

    }
}
