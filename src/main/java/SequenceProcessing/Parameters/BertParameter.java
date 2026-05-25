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

    /**
     * Constructs the parameter bundle for a {@code Bert} model.
     * @param seed The RNG seed shared by weight initialization, shuffling, and masking.
     * @param epoch The number of training epochs.
     * @param optimizer The optimizer used to update the learnable weights.
     * @param initialization The weight initialization strategy.
     * @param loss The loss function attached to the MLM output head.
     * @param wordEmbeddingLength The width of a single token embedding (excluding the bias column).
     * @param multiHeadAttentionLength The number of parallel self-attention heads, {@code N}.
     * @param vocabularyLength The vocabulary size {@code V} (output dimension of the MLM head).
     * @param numEncoderLayers The number of stacked encoder blocks.
     * @param epsilon The numerical-stability constant added inside the LayerNorm denominator.
     * @param feedForwardHiddenLayers The sequence of hidden-layer widths for the position-wise FFN.
     * @param activationFunctions The activation functions applied after each FFN hidden layer.
     * @param gammaValues The per-LayerNorm gamma scale values, consumed in order.
     * @param betaValues The per-LayerNorm beta shift values, consumed in order.
     */
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

    /**
     * Returns the LayerNorm gamma value at the given position in the sequence of values
     * supplied to the constructor.
     * @param index The position to read from.
     * @return The gamma value at {@code index}.
     */
    public double getGammaValue(int index) {
        return gammaValues.get(index);
    }

    /**
     * Returns the LayerNorm beta value at the given position in the sequence of values
     * supplied to the constructor.
     * @param index The position to read from.
     * @return The beta value at {@code index}.
     */
    public double getBetaValue(int index) {
        return betaValues.get(index);
    }

    /**
     * Returns the numerical-stability constant added inside the LayerNorm denominator.
     * @return The {@code epsilon} value.
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Returns the per-head key/query/value dimension, computed as {@code L / N}.
     * @return The dimensionality {@code dk} of a single attention head.
     */
    public int getDk() {
        return L / N;
    }

    /**
     * Returns the biased embedding width, i.e. {@code wordEmbeddingLength + 1}.
     * @return The width {@code L} used throughout the encoder graph.
     */
    public int getL() {
        return L;
    }

    /**
     * Returns the number of parallel self-attention heads.
     * @return The number of heads {@code N}.
     */
    public int getN() {
        return N;
    }

    /**
     * Returns the vocabulary size used by the MLM output head.
     * @return The vocabulary size {@code V}.
     */
    public int getV() {
        return V;
    }

    /**
     * Returns the number of stacked encoder blocks.
     * @return The encoder depth.
     */
    public int getNumEncoderLayers() {
        return numEncoderLayers;
    }

    /**
     * Returns the hidden-layer width at the given depth in the position-wise FFN spec.
     * @param index The hidden-layer position to read from.
     * @return The width of the FFN hidden layer at {@code index}.
     */
    public int getFeedForwardHiddenLayer(int index) {
        return feedForwardHiddenLayers.get(index);
    }

    /**
     * Returns the activation function applied after the FFN hidden layer at the given position.
     * @param index The hidden-layer position to read from.
     * @return The activation function at {@code index}.
     */
    public Object getActivationFunction(int index) {
        return activationFunctions.get(index);
    }

    /**
     * Returns the number of hidden layers in the position-wise feed-forward sub-graph.
     * @return The FFN depth.
     */
    public int getFeedForwardSize() {
        return feedForwardHiddenLayers.size();
    }
}
