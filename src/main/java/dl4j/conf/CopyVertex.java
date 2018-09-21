package dl4j.conf;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by:
 * User: liuzhao
 * Date: 2018/9/20
 * Email: liuzhao@66law.cn
 */
public class CopyVertex extends SameDiffLambdaVertex {

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
        SDVariable in1 = inputs.getInput(0);
        INDArray dup = in1.getArr().dup();
        in1.setArray(dup);
        return in1;
    }

    @Override
    public GraphVertex clone() {
        return new CopyVertex();
    }
}
