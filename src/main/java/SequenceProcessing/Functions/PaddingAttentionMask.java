package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Adds a large-negative bias to attention-score cells that involve padded token
 * positions, so those rows/columns receive ~0 mass under the subsequent Softmax.
 * {@link #validLength} is set per instance before forward (from the real token-row
 * count); when it equals the tensor height, the map is unchanged.
 */
public class PaddingAttentionMask implements Function, Serializable {

    public static final double PADDING_MASK_SENTINEL = -30.0;

    private int validLength;

    public void setValidLength(int validLength) {
        this.validLength = validLength;
    }

    @Override
    public Tensor calculate(Tensor tensor) {
        int rows = tensor.getShape()[0];
        int cols = tensor.getShape()[1];
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = tensor.getValue(new int[]{i, j});
                if (i >= validLength || j >= validLength) {
                    val += PADDING_MASK_SENTINEL;
                }
                values.add(val);
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward;
    }
}
