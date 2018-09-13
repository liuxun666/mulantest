package mulan.util;

import weka.core.matrix.Matrix;

import java.util.Arrays;

public class PageRank {
    private double[][] data;
    private Matrix pr;
    private Matrix pageRankPre;
    private int iterator;

    public PageRank(double[][] data) {
        this.data = data;
    }
    public PageRank(int[][] data) {
        this.data = Arrays.stream(data).map(ds -> Arrays.stream(ds).mapToDouble(d -> (double)d).toArray()).toArray(double[][]::new);
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
            double sum = 0d;
            for (int j = 0; j < PR1.length; j++) {
                sum += PR1[j][i];
            }
            for (int j = 0; j < PR1.length; j++) {
                PR1[j][i] = PR1[j][i] / sum;
            }
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
}