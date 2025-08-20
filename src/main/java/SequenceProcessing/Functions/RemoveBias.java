package SequenceProcessing.Functions;

import ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

import Math.Tensor;

public class RemoveBias implements Function, Serializable {
    @Override
    public Tensor calculate(Tensor matrix) {
        ArrayList<Double> data = (ArrayList<Double>) matrix.getData();
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < data.size() - 1; i++) {
            values.add(data.get(i));
        }
        return new Tensor(values, new int[]{1, values.size()});
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = (ArrayList<Double>) backward.getData();
        ArrayList<Double> newValues = new ArrayList<>(values);
        newValues.add(0.0);
        return new Tensor(newValues, new int[]{1, newValues.size()});
    }
}
