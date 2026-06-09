package SequenceProcessing.Functions;

import ComputationalGraph.Function.Function;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class SelectRow implements Function, Serializable {

    private final int rowIndex;

    public SelectRow(int rowIndex) {
        this.rowIndex = rowIndex;
    }

    @Override
    public Tensor calculate(Tensor tensor) {
        int l = tensor.getShape()[1];
        ArrayList<Double> values = new ArrayList<>();
        for (int j = 0; j < l; j++) {
            values.add(tensor.getValue(new int[]{rowIndex, j}));
        }
        return new Tensor(values, new int[]{1, l});
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        int r = value.getShape()[0];
        int l = value.getShape()[1];
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < l; j++) {
                if (i == rowIndex) {
                    values.add(backward.getValue(new int[]{0, j}));
                } else {
                    values.add(0.0);
                }
            }
        }
        return new Tensor(values, value.getShape());
    }
}
