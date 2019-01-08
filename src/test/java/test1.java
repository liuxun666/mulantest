import mulan.data.MultiLabelInstances;
import mulan.rbms.M;
import mulan.util.StatUtils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class test1 {
    public static void main(String[] args) throws Exception {
        String path = "data";
        String fileName = "emotions";
        String percentage = "80";
        System.out.println("Loading the dataset " + fileName);
        MultiLabelInstances mlDataSet = new MultiLabelInstances(path + "/" + fileName + ".arff", path + "/" +  fileName + ".xml");

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

        MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + "/" + fileName + ".xml");
        MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + "/" + fileName + ".xml");

        int[] labelIndices = train.getLabelIndices();
        Instances dataSet1 = train.getDataSet();

        double[][] labelData = new double[dataSet1.numInstances()][train.getNumLabels()];
        int[][] labelDataInt = new int[dataSet1.numInstances()][train.getNumLabels()];

        for (int i = 0; i < dataSet1.numInstances(); i++){
            for (int j = 0; j < train.getNumLabels(); j++) {
                labelData[i][j] = dataSet1.get(i).value(labelIndices[j]);
                labelDataInt[i][j] = (int)dataSet1.get(i).value(labelIndices[j]);
            }
        }


        double[][] pearsonScore = StatUtils.pearsonScore(labelData);
        double[][] mInfomation = StatUtils.mInfomation(labelDataInt, train.getNumLabels());
        double[][] myMi = StatUtils.margDepMatrix(labelData, train.getNumLabels());
        System.out.println("pearson: \n" + M.toString(pearsonScore));
        System.out.println("mInfomation: \n" + M.toString(mInfomation));
        System.out.println("myMi: \n" + M.toString(myMi));


    }


}
