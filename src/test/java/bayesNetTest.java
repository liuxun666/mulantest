import mulan.data.MultiLabelInstances;
import mulan.util.A;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.estimators.Estimator;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Created by:
 * User: liuzhao
 * Date: 2018/9/17
 * Email: liuzhao@66law.cn
 */
public class bayesNetTest {
    public static void main(String[] args) throws Exception {
        String path = "data\\emotions";
        String filestem = "";
        String percentage = "80";
//
        System.out.println("Loading the dataset");
        MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

        // split the data set into train and test
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

        MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + filestem + ".xml");
        MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + filestem + ".xml");

        BayesNet bayesNet = new BayesNet();
        int[] labelIdx = train.getLabelIndices();
        Instances dataSet1 = train.getDataSet();
        dataSet1.setClassIndex(labelIdx[0]);
        bayesNet.buildClassifier(dataSet1);
        Estimator[][] distributions = bayesNet.getDistributions();

        System.out.println(A.toString(labelIdx));
        for (int i = 0; i < distributions.length; i++) {
            for (int j = 0; j < distributions[i].length; j++) {
                System.out.print(distributions[i][j].getProbability(0) + " ");
            }
            System.out.println("");
        }

    }
}
