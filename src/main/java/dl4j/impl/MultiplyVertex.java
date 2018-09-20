package dl4j.impl;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.graph.vertex.impl.MergeVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Created by:
 * User: liuzhao
 * Date: 2018/9/20
 * Email: liuzhao@66law.cn
 */
public class MultiplyVertex extends MergeVertex {
    public MultiplyVertex(ComputationGraph graph, String name, int vertexIndex) {
        this(graph, name, vertexIndex, null, null);
    }

    public MultiplyVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                       VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
    }

    @Override
    public String toString() {
        return "MultiplyVertex(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\")";
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray first = inputs[0];
        for (int i = 1; i < inputs.length; i++) {
            first = first.mul(inputs[i]);
        }
        return first;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        return super.doBackward(tbptt, workspaceMgr);
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        super.setBackpropGradientsViewArray(backpropGradientsViewArray);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        return super.feedForwardMaskArrays(maskArrays, currentMaskState, minibatchSize);
    }
}
