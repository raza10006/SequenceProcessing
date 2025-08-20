package SequenceProcessing.Parameters;

import Classification.Parameter.LinearPerceptronParameter;
import ComputationalGraph.Function;

import java.io.Serializable;
import java.util.ArrayList;

public class RecurrentNeuralNetworkParameter extends LinearPerceptronParameter implements Serializable {

    private final ArrayList<Integer> hiddenLayers;
    private final ArrayList<Function> functions;
    private final int classLabelSize;

    public RecurrentNeuralNetworkParameter(int seed, double learningRate, double etaDecrease, double crossValidationRatio, int epoch, ArrayList<Integer> hiddenLayers, ArrayList<Function> functions, int classLabelSize) {
        super(seed, learningRate, etaDecrease, crossValidationRatio, epoch);
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

    public void setLearningRate() {
        this.learningRate *= etaDecrease;
    }
}
