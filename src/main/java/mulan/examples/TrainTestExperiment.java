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
package mulan.examples;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import mulan.classifier.transformation.*;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

/**
 * Class demonstrating a simple train/test evaluation experiment
 *
 * @author Grigorios Tsoumakas
 * @version 2012.02.06
 */
public class TrainTestExperiment {


    /**
     * Executes this example
     *
     * @param args command-line arguments -path, -filestem and -percentage 
     * (training set), e.g. -path dataset/ -filestem emotions -percentage 67
     */
    public static void main(String[] args) throws Exception {
        LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
        loggerContext.getLogger("org.nd4j").setLevel(Level.INFO);
//        loggerContext.getLogger("org.deeplearning4j").setLevel(Level.INFO);
//        LoggerContext
        if (args.length < 2){
            System.out.println("useage: java -jar xxxx path filename numlabels batchsize");
            ev("F:\\MyPaper\\数据集\\birds", "birds", 0,512);
            System.exit(0);
        }else if (args.length == 2){
//            ev("H:\\MyPaper\\数据集\\yeast", "yeast");
            ev(args[0], args[1], 0,512);
        }else if (args.length == 3){
            ev(args[0], args[1], 0, Integer.parseInt(args[2]));
        }else{
            ev(args[0], args[1], Integer.parseInt(args[2]), Integer.parseInt(args[3]));
        }


//        try {
//            String result = "result";
//            File resultDir = new File(result);
//            File[] existsFiles = resultDir.listFiles();
//            List<String> existsFileNames = Arrays.stream(existsFiles).map(f -> f.getName().split("\\.")[0]).collect(Collectors.toList());
//            BufferedReader reader = new BufferedReader(new FileReader("numLabels.txt"));
//            String line;
//            Map<String, Integer> dataSetNumLabelsMap = new HashMap<>();
//            while (( line = reader.readLine()) != null){
//                String[] split = line.split("\t");
//                dataSetNumLabelsMap.put(split[0], Integer.valueOf(split[1]));
//            }
//            dataSetNumLabelsMap = sortByValue(dataSetNumLabelsMap);
//            for (String existsFileName : existsFileNames) {
//                dataSetNumLabelsMap.remove(existsFileName);
//            }
//            dataSetNumLabelsMap.forEach((k, v) -> System.out.println(k + "\t" + v));
//            Map<String, String> dataSetParentMap = new HashMap<>();
//            String path = "data";
//            File file = new File(path);
//            File[] subDirs = file.listFiles();
//            for (File dir : subDirs) {
//                File[] files = dir.listFiles(new FilenameFilter() {
//                    @Override
//                    public boolean accept(File dir, String name) {
//                        if (existsFileNames.contains(name) || name.contains(".xml") || name.contains("test") || name.contains("train") || name.contains("README")) {
//                            return false;
//                        } else {
//                            return true;
//                        }
//                    }
//                });
//                for (File f : files) {
//                    String fpath = f.getParent();
//                    String[] split = f.getName().split("\\.");
//                    String filename = split[0];
//                    dataSetParentMap.put(filename, fpath);
//                }
//            }
//
//            dataSetNumLabelsMap.forEach((k, v) -> {
//                String pPath = dataSetParentMap.get(k);
//                if(pPath != null){
//                    try{
//                        ev(pPath, k);
//                    }catch (Exception e){
//                        e.printStackTrace();
//                    }
//                }else{
//                    System.out.println(k + " 没有找到目录");
//                }
//            });
//
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    }

    public static void ev(String path, String fileName, int numLabels, int batchSize) throws Exception {
        String percentage = "80";
        System.out.println("Loading the dataset " + fileName);
        MultiLabelInstances train;
        MultiLabelInstances test;
        if(numLabels == 0){
            train = new MultiLabelInstances(path + "/" + fileName + "-train.arff", path + "/" +  fileName + ".xml");
            test = new MultiLabelInstances(path + "/" + fileName + "-test.arff", path + "/" +  fileName + ".xml");
        }else{
            train = new MultiLabelInstances(path + "/" + fileName + "-train.arff", numLabels);
            test = new MultiLabelInstances(path + "/" + fileName + "-test.arff", numLabels);
        }

//        Instances dataSet = mlDataSet.getDataSet();
//        RemovePercentage rmvp = new RemovePercentage();
//        rmvp.setInvertSelection(true);
//        rmvp.setPercentage(Double.parseDouble(percentage));
//        rmvp.setInputFormat(dataSet);
//        Instances trainDataSet = Filter.useFilter(dataSet, rmvp);
//
//        RemoveType removeType = new RemoveType();
//        removeType.setInputFormat(trainDataSet);
//        trainDataSet = Filter.useFilter(trainDataSet, removeType);
//
//        rmvp = new RemovePercentage();
//        rmvp.setPercentage(Double.parseDouble(percentage));
//        rmvp.setInputFormat(dataSet);
//        Instances testDataSet = Filter.useFilter(dataSet, rmvp);
//
//        removeType.setInputFormat(testDataSet);
//        testDataSet = Filter.useFilter(testDataSet, removeType);
//
//        MultiLabelInstances train = new MultiLabelInstances(trainDataSet, path + "/" + fileName + ".xml");
//        MultiLabelInstances test = new MultiLabelInstances(testDataSet, path + "/" + fileName + ".xml");

        Evaluator eval = new Evaluator();
        Evaluation results1;
        Evaluation results2;
        Evaluation results3;
        Evaluation results4;
        Evaluation results5;

        Classifier brClassifier = new SMO();
//        AttMiPageRankNeuralNet acc = new AttMiPageRankNeuralNet(brClassifier, brClassifier, batchSize);
        BinaryRelevance br = new BinaryRelevance(brClassifier);
        MultiLabelStacking mbr = new MultiLabelStacking(brClassifier, brClassifier);
        ClassifierChain cc = new ClassifierChain(brClassifier);
        PageRankMiClassifierChain pcc = new PageRankMiClassifierChain(brClassifier);
//        ClassifierChain_bak cc = new ClassifierChain_bak(brClassifier);
//        EnsembleOfClassifierChains ecc = new EnsembleOfClassifierChains(brClassifier, 10, true, true);
        PageRankDoubleLayerCC mcc = new PageRankDoubleLayerCC(brClassifier, brClassifier);

        br.setDebug(true);
        mbr.setDebug(true);
        cc.setDebug(true);
        pcc.setDebug(true);
        mcc.setDebug(true);
//        acc.setDebug(true);

        PrintWriter writer = new PrintWriter(new FileWriter("result/" + fileName + ".txt"));

        MultiLabelInstances clone = train.clone();
        System.out.println("==============================mbr================================");
        mbr.build(clone);
        results2 = eval.evaluate(mbr, test, train);
        mbr = null;
        writer.println("evaluate " + path + "/" + fileName);
        writer.println("mbr:\n" + results2);
        writer.flush();

        System.out.println("==============================br================================");
        clone = train.clone();
        br.build(clone);
        results1 = eval.evaluate(br, test, train);
        br = null;
        writer.println("br:\n" + results1);
        writer.flush();

        clone = train.clone();
        System.out.println("==============================cc================================");
        cc.build(clone);
        results3 = eval.evaluate(cc, test, train);
        cc = null;
        writer.println("cc:\n" + results3);
        writer.flush();

        System.out.println("==============================pcc================================");
        pcc.build(clone);
        results3 = eval.evaluate(pcc, test, train);
        pcc = null;
        writer.println("pcc:\n" + results3);
        writer.flush();

        clone = train.clone();
        System.out.println("==============================mcc================================");
        mcc.build(clone);
        results4 = eval.evaluate(mcc, test, train);
        mcc = null;
        writer.println("mcc:\n" + results4);
        writer.flush();
        writer.close();
//
//
//        clone = train.clone();
//        System.out.println("==============================acc================================");
//        acc.build(clone);
//        results5 = eval.evaluate(acc, test, train);
//        acc = null;
//        writer.println("acc:\n" + results5);
//        writer.flush();
//        writer.close();

        System.out.println("done！");

//        results5 = eval.evaluate(acc, test, train);
//        writer.println("evaluate " + path + "/" + fileName);
//        writer.println("br:\n" + results1);
//        writer.println("mbr:\n" + results2);
//        writer.println("cc:\n" + results3);
//        writer.println("mcc:\n" + results4);
////        writer.println("acc:\n" + results5);
//        writer.flush();
//        writer.close();

    }

    public static Map sortByValue(Map map) {
        List list = new LinkedList(map.entrySet());
        Collections.sort(list, new Comparator() {

            public int compare(Object o1, Object o2) {
                return ((Comparable) ((Map.Entry) (o1)).getValue())
                        .compareTo(((Map.Entry) (o2)).getValue());

            }
        });
        Map result = new LinkedHashMap();

        for (Iterator it = list.iterator(); it.hasNext();) {
            Map.Entry entry = (Map.Entry) it.next();
            result.put(entry.getKey(), entry.getValue());
        }
        return result;
    }

}