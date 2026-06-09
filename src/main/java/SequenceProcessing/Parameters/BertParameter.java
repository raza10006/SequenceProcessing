package SequenceProcessing.Parameters;

import ComputationalGraph.Initialization.Initialization;

import java.io.Serializable;
import java.util.ArrayList;

public class BertParameter extends TransformerParameter implements Serializable {

    private final int numEncoderLayers;

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
        // BERT keeps a single set of layer / activation / gamma / beta values, whereas the parent
        // TransformerParameter models a separate encoder (input) and decoder (output) split. We
        // therefore pass the single BERT list for BOTH the input and output parameters so the parent
        // is fully initialized and every inherited getter returns BERT's values. The shared scalars
        // (wordEmbeddingLength, multiHeadAttentionLength, vocabularyLength, epsilon) map one-to-one;
        // the parent derives L = wordEmbeddingLength + 1 just as BERT did before.
        super(seed, epoch, optimizer, initialization, loss,
                wordEmbeddingLength, multiHeadAttentionLength, vocabularyLength, epsilon,
                feedForwardHiddenLayers, feedForwardHiddenLayers,
                activationFunctions, activationFunctions,
                gammaValues, gammaValues,
                betaValues, betaValues);
        // numEncoderLayers is BERT-native (no equivalent on TransformerParameter), so it stays here.
        this.numEncoderLayers = numEncoderLayers;
    }

    /**
     * Returns the LayerNorm gamma value at the given position in the sequence of values
     * supplied to the constructor.
     * @param index The position to read from.
     * @return The gamma value at {@code index}.
     */
    public double getGammaValue(int index) {
        // BERT uses one gamma list; delegate to the parent's input-side accessor, which holds
        // that same list (passed for both input and output in the constructor).
        return getGammaInputValue(index);
    }

    /**
     * Returns the LayerNorm beta value at the given position in the sequence of values
     * supplied to the constructor.
     * @param index The position to read from.
     * @return The beta value at {@code index}.
     */
    public double getBetaValue(int index) {
        // BERT uses one beta list; delegate to the parent's input-side accessor, which holds
        // that same list (passed for both input and output in the constructor).
        return getBetaInputValue(index);
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
        // BERT's single FFN spec is stored as the parent's input-side hidden-layer list.
        return getInputHiddenLayer(index);
    }

    /**
     * Returns the activation function applied after the FFN hidden layer at the given position.
     * @param index The hidden-layer position to read from.
     * @return The activation function at {@code index}.
     */
    public Object getActivationFunction(int index) {
        // BERT's single FFN spec is stored as the parent's input-side activation list.
        return getInputActivationFunction(index);
    }

    /**
     * Returns the number of hidden layers in the position-wise feed-forward sub-graph.
     * @return The FFN depth.
     */
    public int getFeedForwardSize() {
        // BERT's single FFN spec is stored as the parent's input-side hidden-layer list.
        return getInputSize();
    }
}
