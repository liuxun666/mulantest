import mulan.data.MultiLabelInstances;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class test1 {
    public static void main(String[] args) throws Exception {
        String path = "F:\\code\\mulan-master\\data\\multi-label\\birds\\";
        String filestem = "birds";
        String percentage = "80";
//
        System.out.println("Loading the dataset");
        MultiLabelInstances mlDataSet = new MultiLabelInstances(path + filestem + "-train.arff", path + filestem + ".xml");

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
        Instances instances = train.getDataSet();
        instances.setClassIndex(train.getLabelIndices()[0]);
        bayesNet.buildClassifier(instances);
        int numNodes = bayesNet.getNrOfNodes();
        for (int i = 0; i < numNodes; i++) {
//            System.out.println(bayesNet.getNrOfParents(i));
            for (int j = 0; j < bayesNet.getCardinality(i); j++) {
                System.out.println(j);
//                for (int k = 0; k < bayesNet.getParentCardinality(i); k++) {
//                    System.out.println(bayesNet.getNodeValue(i, j) + "=" + bayesNet.getProbability(i, k, j) );
//                }
            }
        }

    }
}
