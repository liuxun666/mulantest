package dl4j.conf;

import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Created by:
 * User: liuzhao
 * Date: 2018/9/20
 * Email: liuzhao@66law.cn
 */
public class MultiplyVertex extends SameDiffLambdaVertex {

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
        SDVariable in1 = inputs.getInput(0);
        SDVariable in2 = inputs.getInput(1);
        SDVariable ret = in1.mul(in2);
        return ret;
    }

    @Override
    public GraphVertex clone() {
        return new MultiplyVertex();
    }
}
