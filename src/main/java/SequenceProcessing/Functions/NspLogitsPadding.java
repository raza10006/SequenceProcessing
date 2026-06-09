package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class NspLogitsPadding implements Function, Serializable {

    public static final double NSP_PADDING_SENTINEL = -30.0;

    private final int vocabularySize;

    public NspLogitsPadding(int vocabularySize) {
        this.vocabularySize = vocabularySize;
    }

    @Override
    public Tensor calculate(Tensor tensor) {
        ArrayList<Double> values = new ArrayList<>();
        values.add(tensor.getValue(new int[]{0, 0}));
        values.add(tensor.getValue(new int[]{0, 1}));
        for (int j = 2; j < vocabularySize; j++) {
            values.add(NSP_PADDING_SENTINEL);
        }
        return new Tensor(values, new int[]{1, vocabularySize});
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = new ArrayList<>();
        values.add(backward.getValue(new int[]{0, 0}));
        values.add(backward.getValue(new int[]{0, 1}));
        return new Tensor(values, new int[]{1, 2});
    }
}
