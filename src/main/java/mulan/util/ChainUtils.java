package mulan.util;

import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import mulan.data.MultiLabelInstances;
import mulan.rbms.M;

import java.util.Arrays;
import java.util.stream.Stream;

/**
 * Created by:
 * User: liuzhao
 * Date: 2018/10/10
 * Email: liuzhao@66law.cn
 */
public class ChainUtils {
    public static int[] getMiCCChain(MultiLabelInstances train) {
        Stream<int[]> list = train.getDataSet().stream()
                .map(instance -> Arrays.stream(train.getLabelIndices()).map(i -> (int)instance.value(i)).toArray());
        int[][] labels = list.toArray(int[][]::new);

        PageRank pageRank = new PageRank(labels).build();
        double[] pr = pageRank.getPr().getColumnPackedCopy();
//        double[] pr = pageRank(labels);
        System.out.println("pr" + Arrays.toString(pr));
        int root = maxIndex(pr);
//        int root = minIndex(pr);
//        double[][] ud = StatUtils.margDepMatrix(labels, train.getNumLabels());
        double[][] ud = StatUtils.mInfomation(labels, train.getNumLabels());
        System.out.println("互信息矩阵 ：");
        System.out.println(M.toString(ud));
        EdgeWeightedGraph G = new EdgeWeightedGraph(train.getNumLabels());

        for(int i = 0; i < train.getNumLabels(); i++) {
            for(int j = i+1; j < train.getNumLabels(); j++) {
                Edge e = new Edge(i, j, -ud[i][j]);
                G.addEdge(e);
            }
        }

        KruskalMST mst = new KruskalMST(G);
        int paM[][] = new int[train.getNumLabels()][train.getNumLabels()];
        for (Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paM[j][k] = 1;
            paM[k][j] = 1;
        }

        System.out.println("Make a Tree from Root " + root);
        //这里paL[][]被初始化了，那么里面的元素就是全部0；
        int paL[][] = new int[train.getNumLabels()][0];
        int visted[] = new int[train.getNumLabels()];
        Arrays.fill(visted,-1);
        visted[root] = 0;
        //treeify(int:root=0树的根节点,int[标签数目][标签数目]：paM存储的是标签之间是否有边，有边则标签号交汇处值为1，int[标签数][0]：0，int[标签数]：visited记录标签是否被访问，[0，-1，-1，-1，-1，-1] )
        treeify(root,paM,paL, visted);
        return visted;
    }

    public static int[] getPearsonCCChain(MultiLabelInstances train) {
        Stream<double[]> list = train.getDataSet().stream()
                .map(instance -> Arrays.stream(train.getLabelIndices()).mapToDouble(instance::value).toArray());
        double[][] labels = list.toArray(double[][]::new);

        PageRank pageRank = new PageRank(labels).build();
        double[] pr = pageRank.getPr().getColumnPackedCopy();
//        double[] pr = pageRank(labels);
        System.out.println("pr" + Arrays.toString(pr));
        int root = maxIndex(pr);
//        int root = minIndex(pr);
//        double[][] ud = StatUtils.margDepMatrix(labels, train.getNumLabels());
        double[][] ud = StatUtils.pearsonScore(labels);
        System.out.println("互信息矩阵 ：");
        System.out.println(M.toString(ud));
        EdgeWeightedGraph G = new EdgeWeightedGraph(train.getNumLabels());

        for(int i = 0; i < train.getNumLabels(); i++) {
            for(int j = i+1; j < train.getNumLabels(); j++) {
                Edge e = new Edge(i, j, -ud[i][j]);
                G.addEdge(e);
            }
        }

        KruskalMST mst = new KruskalMST(G);
        int paM[][] = new int[train.getNumLabels()][train.getNumLabels()];
        for (Edge e : mst.edges()) {
            int j = e.either();
            int k = e.other(j);
            paM[j][k] = 1;
            paM[k][j] = 1;
        }

        System.out.println("Make a Tree from Root " + root);
        //这里paL[][]被初始化了，那么里面的元素就是全部0；
        int paL[][] = new int[train.getNumLabels()][0];
        int visted[] = new int[train.getNumLabels()];
        Arrays.fill(visted,-1);
        visted[root] = 0;
        //treeify(int:root=0树的根节点,int[标签数目][标签数目]：paM存储的是标签之间是否有边，有边则标签号交汇处值为1，int[标签数][0]：0，int[标签数]：visited记录标签是否被访问，[0，-1，-1，-1，-1，-1] )
        treeify(root,paM,paL, visted);
        return visted;
    }

    public static void treeify(int root, int paM[][], int paL[][], int visited[]) {
        int children[] = new int[]{};
        for(int j = 0; j < paM[root].length; j++) {
            if (paM[root][j] == 1) {
                if (visited[j] < 0) {
                    //A表示是一个方便数组操作的类，该处调用了该类的add方法
                    children = A.add(children,j);
                    paL[j] = A.add(paL[j],root);
                    //返回数组visited[]中最大元素的索引号
                    visited[j] = visited[maxIndex(visited)] + 1;
                }//endif
                // set as visited
                //paM[root][j] = 0;
            }//end1if
        }//endfor
        // go through again
        for(int child : children) {
            treeify(child,paM,paL,visited);
        }
    }

    public static int maxIndex(int[] a) {
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

    public static int maxIndex(double[] a) {
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
}
