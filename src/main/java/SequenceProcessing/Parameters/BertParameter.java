package SequenceProcessing.Parameters;

import ComputationalGraph.Initialization.Initialization;
import ComputationalGraph.NeuralNetworkParameter;

import java.io.Serializable;
import java.util.ArrayList;

public class BertParameter extends NeuralNetworkParameter implements Serializable {

    private final int L;
    private final int N;
    private final int V;
    private final int numEncoderLayers;
    private final double epsilon;
    private final ArrayList<Integer> feedForwardHiddenLayers;
    private final ArrayList<Object> activationFunctions;
    private final ArrayList<Double> gammaValues;
    private final ArrayList<Double> betaValues;

    public BertParameter(int seed, int epoch, ComputationalGraph.Optimizer.Optimizer optimizer, Initialization initialization, ComputationalGraph.Loss.Loss loss, int wordEmbeddingLength, int multiHeadAttentionLength, int vocabularyLength, int numEncoderLayers, double epsilon, ArrayList<Integer> feedForwardHiddenLayers, ArrayList<Object> activationFunctions, ArrayList<Double> gammaValues, ArrayList<Double> betaValues) {
        super(seed, epoch, optimizer, initialization, loss, 0.0, -1);
        this.L = wordEmbeddingLength + 1;
        this.N = multiHeadAttentionLength;
        this.V = vocabularyLength;
        this.numEncoderLayers = numEncoderLayers;
        this.epsilon = epsilon;
        this.feedForwardHiddenLayers = feedForwardHiddenLayers;
        this.activationFunctions = activationFunctions;
        this.gammaValues = gammaValues;
        this.betaValues = betaValues;
    }

    public double getGammaValue(int index) {
        return gammaValues.get(index);
    }

    public double getBetaValue(int index) {
        return betaValues.get(index);
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

    public int getNumEncoderLayers() {
        return numEncoderLayers;
    }

    public int getFeedForwardHiddenLayer(int index) {
        return feedForwardHiddenLayers.get(index);
    }

    public Object getActivationFunction(int index) {
        return activationFunctions.get(index);
    }

    public int getFeedForwardSize() {
        return feedForwardHiddenLayers.size();
    }
}
