package SequenceProcessing.Parameters;

import ComputationalGraph.Function.Function;
import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.NeuralNetworkParameter;

import java.io.Serializable;
import java.util.ArrayList;

public class TransformerParameter extends NeuralNetworkParameter implements Serializable {

    private final int L;
    private final int N;
    private final int V;
    private final double epsilon;
    private final ArrayList<Integer> inputHiddenLayers;
    private final ArrayList<Integer> outputHiddenLayers;
    private final ArrayList<ComputationalGraph.Function.Function> inputFunctions;
    private final ArrayList<ComputationalGraph.Function.Function> outputFunctions;
    private final ArrayList<Double> gammaInputValues;
    private final ArrayList<Double> gammaOutputValues;
    private final ArrayList<Double> betaInputValues;
    private final ArrayList<Double> betaOutputValues;

    public TransformerParameter(int seed, int epoch, ComputationalGraph.Optimizer.Optimizer optimizer, Initialization initialization, int wordEmbeddingLength, int multiHeadAttentionLength, int vocabularyLength, double epsilon, ArrayList<Integer> inputHiddenLayers, ArrayList<Integer> outputHiddenLayers, ArrayList<Function> inputActivationFunctions, ArrayList<Function> outputActivationFunctions, ArrayList<Double> gammaInputValues, ArrayList<Double> gammaOutputValues, ArrayList<Double> betaInputValues, ArrayList<Double> betaOutputValues) {
        super(seed, epoch, optimizer, initialization, 0.0);
        this.L = wordEmbeddingLength + 1;
        this.N = multiHeadAttentionLength;
        this.V = vocabularyLength;
        this.epsilon = epsilon;
        this.inputHiddenLayers = inputHiddenLayers;
        this.outputHiddenLayers = outputHiddenLayers;
        this.inputFunctions = inputActivationFunctions;
        this.outputFunctions = outputActivationFunctions;
        this.gammaInputValues = gammaInputValues;
        this.gammaOutputValues = gammaOutputValues;
        this.betaInputValues = betaInputValues;
        this.betaOutputValues = betaOutputValues;
    }

    public double getGammaInputValue(int index) {
        return  gammaInputValues.get(index);
    }

    public double getGammaOutputValue(int index) {
        return gammaOutputValues.get(index);
    }

    public double getBetaInputValue(int index) {
        return betaInputValues.get(index);
    }

    public double getBetaOutputValue(int index) {
        return betaOutputValues.get(index);
    }

    public double getEpsilon() {
        return epsilon;
    }

    public int getDk() {
        return L / N;
    }

    public int getL() {
        return L;
    }

    public int getN() {
        return N;
    }

    public int getV() {
        return V;
    }

    public int getInputHiddenLayer(int index) {
        return inputHiddenLayers.get(index);
    }

    public int getOutputHiddenLayer(int index) {
        return outputHiddenLayers.get(index);
    }

    public Function getInputActivationFunction(int index) {
        return inputFunctions.get(index);
    }

    public Function getOutputActivationFunction(int index) {
        return outputFunctions.get(index);
    }

    public int getInputSize() {
        return inputHiddenLayers.size();
    }

    public int getOutputSize() {
        return outputHiddenLayers.size();
    }
}
