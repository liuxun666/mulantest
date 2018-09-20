package mulan.util;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by:
 * User: liuzhao
 * Date: 2018/9/20
 * Email: liuzhao@66law.cn
 */
public class Attention {
    public INDArray W;
    public INDArray b;
    public Attention(INDArray W, INDArray b){
        this.W = W;
        this.b = b;
    }
}
