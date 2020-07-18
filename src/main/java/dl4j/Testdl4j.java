package dl4j;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Testdl4j {
    public static void main(String[] args) throws Exception {
        LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
        loggerContext.getLogger("org.nd4j").setLevel(Level.INFO);
        if (args.length < 2){
            System.out.println("useage: java -jar xxxx path filename");
            ev("F:\\MyPaper\\数据集\\birds", "birds", 0);
            System.exit(0);
        }else if (args.length == 2){
//            ev("H:\\MyPaper\\数据集\\yeast", "yeast");
            ev(args[0], args[1], 0);
        }else{
            ev(args[0], args[1], Integer.parseInt(args[2]));
        }
    }

    private static void ev(String path, String fileName, int numLabels) throws Exception {
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

        MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + "/" + fileName + ".xml");
        MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + "/" + fileName + ".xml");

        Evaluator eval = new Evaluator();
        Evaluation results5;

        Classifier brClassifier = new SMO();
        TrainDl4j acc = new TrainDl4j(brClassifier, brClassifier);
        acc.setDebug(true);

        MultiLabelInstances clone = train.clone();
        System.out.println("==============================acc================================");
        acc.build(clone);


        results5 = eval.evaluate(acc, test, train);
        PrintWriter writer = new PrintWriter(new FileWriter("result/" + fileName + "_TestDl4j.txt"));
        writer.println("evaluate " + path + "/" + fileName);

        writer.println("acc:\n" + results5);
        writer.flush();
        writer.close();

    }
}
