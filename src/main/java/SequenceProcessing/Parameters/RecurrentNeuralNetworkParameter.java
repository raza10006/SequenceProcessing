package SequenceProcessing.Parameters;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.NeuralNetworkParameter;

import java.io.Serializable;
import java.util.ArrayList;

public class RecurrentNeuralNetworkParameter extends NeuralNetworkParameter implements Serializable {

    private final ArrayList<Integer> hiddenLayers;
    private final ArrayList<Function> functions;
    private final int classLabelSize;

    public RecurrentNeuralNetworkParameter(int seed, int epoch, ComputationalGraph.Optimizer.Optimizer optimizer, Initialization initialization, ArrayList<Integer> hiddenLayers, ArrayList<Function> functions, int classLabelSize) {
        super(seed, epoch, optimizer, initialization, 0.0);
        this.hiddenLayers = hiddenLayers;
        this.functions = functions;
        this.classLabelSize = classLabelSize;
    }

    public int size() {
        return hiddenLayers.size();
    }

    public int getClassLabelSize() {
        return classLabelSize;
    }

    public Function getActivationFunction(int index) {
        return functions.get(index);
    }

    public Integer getHiddenLayer(int index) {
        return hiddenLayers.get(index);
    }
}
