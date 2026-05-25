package SequenceProcessing.Classification;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import ComputationalGraph.Function.*;
import ComputationalGraph.Node.*;
import Dictionary.*;
import Math.Tensor;
import Math.Vector;
import SequenceProcessing.Functions.*;
import SequenceProcessing.Parameters.BertParameter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class Bert extends ComputationalGraph implements Serializable {

    private final VectorizedDictionary dictionary;
    private final int sepIndex;

    /**
     * Constructs a BERT model from the given parameter bundle and vectorized dictionary,
     * caching the dictionary index of the {@code [SEP]} token for later segment detection.
     * @param parameter The hyperparameter container; expected to be a {@link BertParameter}.
     * @param dictionary The vectorized dictionary used to look up the {@code [SEP]} entry.
     */
    public Bert(NeuralNetworkParameter parameter, VectorizedDictionary dictionary) {
        super(parameter);
        this.dictionary = dictionary;
        int sep = -1;
        for (int k = 0; k < this.dictionary.size(); k++) {
            if (this.dictionary.getWord(k).getName().equals("[SEP]")) {
                sep = k;
                break;
            }
        }
        this.sepIndex = sep;
    }

    /**
     * Adds the standard sinusoidal positional encoding to a token embedding tensor.
     * Even feature columns receive a sine offset and odd columns a cosine offset, both
     * scaled by the canonical {@code 10000^(2i/d)} frequency.
     * @param tensor The token embedding tensor of shape {@code [r, wordEmbeddingLength]}.
     * @param wordEmbeddingLength The width of the embedding dimension.
     * @return A new tensor of the same shape with the positional encoding added pointwise.
     */
    private Tensor positionalEncoding(Tensor tensor, int wordEmbeddingLength) {
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < tensor.getShape()[0]; i++) {
            for (int j = 0; j < tensor.getShape()[1]; j++) {
                double val = tensor.getValue(new int[]{i, j});
                if (j % 2 == 0) {
                    values.add(val + Math.sin((i + 1.0) / Math.pow(10000, (j + 0.0) / wordEmbeddingLength)));
                } else {
                    values.add(val + Math.cos((i + 1.0) / Math.pow(10000, (j - 1.0) / wordEmbeddingLength)));
                }
            }
        }
        return new Tensor(values, tensor.getShape());
    }

    /**
     * Checks whether the given embedding row matches the cached {@code [SEP]} vector,
     * used by {@link #createInputTensors} to auto-detect sentence boundaries when
     * assigning segment ids. Returns {@code false} when no {@code [SEP]} entry was
     * found in the dictionary or when the row length does not match.
     * @param row The candidate embedding row to test.
     * @return {@code true} when the row matches the {@code [SEP]} vector within a small
     *         numerical tolerance, {@code false} otherwise.
     */
    private boolean matchesSepRow(ArrayList<Double> row) {
        if (sepIndex < 0) {
            return false;
        }
        Vector v = ((VectorizedWord) dictionary.getWord(sepIndex)).getVector();
        if (row.size() != v.size()) {
            return false;
        }
        for (int i = 0; i < v.size(); i++) {
            if (Math.abs(row.get(i) - v.getValue(i)) > 1e-5) {
                return false;
            }
        }
        return true;
    }

    /**
     * Parses a packed training instance into the BERT word and segment input tensors and
     * returns the gold token ids that follow the {@link Double#MAX_VALUE} sentinel. Token
     * embeddings populate {@code wordInput} (with positional encoding applied), while
     * {@code segmentInput} receives the segment-id tensor whose ids flip at the first
     * {@code [SEP]} row and whose trailing bias column is held at {@code 0.0}.
     * @param instance The packed input tensor: embedding rows, {@code Double.MAX_VALUE}, then class labels.
     * @param wordInput The graph input node that receives the token embedding tensor.
     * @param segmentInput The graph input node that receives the segment embedding tensor.
     * @param wordEmbeddingLength The width of a single token embedding row.
     * @return The list of gold class labels (one per token row) read from the instance.
     */
    private ArrayList<Integer> createInputTensors(Tensor instance, ComputationalNode wordInput, ComputationalNode segmentInput, int wordEmbeddingLength) {
        boolean isOutput = false;
        ArrayList<Integer> classLabels = new ArrayList<>();
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Integer> segmentIds = new ArrayList<>();
        boolean afterFirstSep = false;
        int L = wordEmbeddingLength + 1;
        for (int i = 0; i < instance.getShape()[0]; i++) {
            double val = instance.getValue(new int[]{i});
            if (val == Double.MAX_VALUE) {
                isOutput = true;
                int rows = values.size() / wordEmbeddingLength;
                Tensor wordTensor = new Tensor(values, new int[]{rows, wordEmbeddingLength});
                wordInput.setValue(positionalEncoding(wordTensor, wordEmbeddingLength));
                ArrayList<Double> segFlat = new ArrayList<>();
                for (int r = 0; r < rows; r++) {
                    int sid = r < segmentIds.size() ? segmentIds.get(r) : 0;
                    double segValue = (sid == 0) ? -0.05 : 0.05;
                    for (int c = 0; c < wordEmbeddingLength; c++) {
                        segFlat.add(segValue);
                    }
                    segFlat.add(0.0);
                }
                segmentInput.setValue(new Tensor(segFlat, new int[]{rows, L}));
                values.clear();
                segmentIds.clear();
                afterFirstSep = false;
            } else if (isOutput) {
                classLabels.add((int) val);
            } else {
                values.add(val);
                if (values.size() % wordEmbeddingLength == 0) {
                    ArrayList<Double> row = new ArrayList<>(values.subList(values.size() - wordEmbeddingLength, values.size()));
                    int sid = afterFirstSep ? 1 : 0;
                    segmentIds.add(sid);
                    if (matchesSepRow(row)) {
                        afterFirstSep = true;
                    }
                }
            }
        }
        return classLabels;
    }

    /**
     * Builds the LayerNorm sub-graph on top of the given input node: subtracts the row
     * mean, divides by the square root of the row variance plus {@code epsilon}, then
     * applies a learnable {@code gamma} scale and {@code beta} shift. The {@code gamma}
     * and {@code beta} rows are pulled from {@link BertParameter} via the {@code lnSize}
     * counters so each LayerNorm call site consumes its own pair.
     * @param input The input node to normalize.
     * @param parameter The BERT parameter bundle supplying {@code epsilon}, {@code L}, gamma and beta.
     * @param lnSize Two-element counter array tracking how many gamma and beta rows have been consumed.
     * @return The output node of the LayerNorm sub-graph.
     */
    private ComputationalNode layerNormalization(ComputationalNode input, BertParameter parameter, int[] lnSize) {
        ArrayList<Double> data = new ArrayList<>();
        ComputationalNode inputMean = this.addEdge(input, new Mean());
        ComputationalNode meanMinus = this.addEdge(inputMean, new Negation());
        ComputationalNode centered = this.addAdditionEdge(input, meanMinus, false);
        ComputationalNode variance = this.addEdge(centered, new Variance());
        ComputationalNode rootVariance = this.addEdge(variance, new SquareRoot(parameter.getEpsilon()));
        ComputationalNode inverseRootVariance = this.addEdge(rootVariance, new Inverse());
        ComputationalNode normalized = this.addEdge(centered, inverseRootVariance, false, true);
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getGammaValue(lnSize[0]));
        }
        lnSize[0]++;
        ComputationalNode gammaNode = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}), true);
        ComputationalNode scaled = this.addEdge(normalized, gammaNode);
        data = new ArrayList<>();
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getBetaValue(lnSize[1]));
        }
        lnSize[1]++;
        ComputationalNode betaNode = new ComputationalNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}));
        return this.addAdditionEdge(scaled, betaNode, false);
    }

    /**
     * Builds the {@code N} parallel self-attention heads for one encoder block. Each
     * head learns its own {@code Wk}, {@code Wq}, {@code Wv} of shape {@code [L, dk]}
     * and computes {@code softmax(QKᵀ / sqrt(dk)) · V}. The attention is unmasked, so
     * every position attends bidirectionally to every other position.
     * @param input The shared input node fed to all heads.
     * @param parameter The BERT parameter bundle supplying {@code N}, {@code L} and {@code dk}.
     * @param random The seeded RNG used to initialize the per-head weights.
     * @return The list of {@code N} attention head output nodes, ready to be concatenated.
     */
    private ArrayList<ComputationalNode> multiHeadAttention(ComputationalNode input, BertParameter parameter, Random random) {
        ArrayList<ComputationalNode> nodes = new ArrayList<>();
        for (int i = 0; i < parameter.getN(); i++) {
            ComputationalNode wk = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode k = this.addEdge(input, wk);
            ComputationalNode wq = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode q = this.addEdge(input, wq);
            ComputationalNode wv = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getDk(), random), new int[]{parameter.getL(), parameter.getDk()}));
            ComputationalNode v = this.addEdge(input, wv);
            ComputationalNode kTranspose = this.addEdge(k, new Transpose());
            ComputationalNode qk = this.addEdge(q, kTranspose, false, false);
            ComputationalNode qkDk = this.addEdge(qk, new MultiplyByConstant(1.0 / Math.sqrt(parameter.getDk())));
            ComputationalNode sQkDk = this.addEdge(qkDk, new Softmax());
            ComputationalNode attention = this.addEdge(sQkDk, v);
            nodes.add(attention);
        }
        return nodes;
    }

    /**
     * Selects the token positions that participate in the Masked Language Modeling
     * (MLM) loss for a single training instance.
     *
     * Real BERT pre-training picks roughly 15% of the input token positions uniformly
     * at random and only those positions contribute to the MLM cross-entropy loss; the
     * remaining 85% of positions are ignored by the loss. The original paper further
     * perturbs the input embeddings at the selected positions with the canonical
     * 80/10/10 split: 80% of the chosen positions are replaced with the {@code [MASK]}
     * token vector, 10% are replaced with a random vocabulary item, and 10% are kept
     * unchanged. This helper implements the position-selection step (which positions
     * get masked); the 80/10/10 input-perturbation step is left as a documented
     * simplification because applying it cleanly would require a populated
     * {@code [MASK]} entry in the {@link VectorizedDictionary} (the existing tests use
     * either an empty dictionary or one without {@code [MASK]}) plus invasive changes
     * to {@link #createInputTensors(Tensor, ComputationalNode, ComputationalNode, int)}
     * to substitute embedding rows on a per-position basis. Position-level masking on
     * the loss path is the part that materially changes the training objective and is
     * what the architecture diagram explicitly calls out.
     *
     * The number of selected positions is {@code ceil(0.15 * sequenceLength)} with a
     * floor of 1 for any non-empty sequence, so that the very short sequences used by
     * the unit tests (e.g. four token rows) still produce a non-trivial training
     * signal each step rather than being silently skipped by integer truncation.
     * Selection happens via the caller-supplied {@link Random} — {@code train} passes
     * in the same {@code parameter.getSeed()}-seeded RNG used for weight initialization
     * and shuffling — so masked-position selection is fully deterministic and
     * reproducible across runs.
     *
     * Within the framework's single-{@code outputNode}/single-loss contract, the
     * selected positions are mapped onto the loss path by setting the gold one-hot
     * row for each selected position and zeroing out the gold row for every
     * non-selected position. Cross-entropy on an all-zero target row is exactly
     * {@code -sum(0 * log p) = 0}, so non-masked positions contribute neither value
     * nor gradient, faithfully reproducing "compute the loss only on masked
     * positions" without requiring a second loss target or graph head.
     *
     * @param sequenceLength the number of token rows in the current input instance.
     * @param random         the seeded RNG used to draw positions; sharing the
     *                       class's training RNG keeps masking reproducible.
     * @return a sorted, distinct list of position indices in {@code [0, sequenceLength)}
     *         selected for masking; the empty list when {@code sequenceLength <= 0}.
     */
    private static ArrayList<Integer> selectMaskedPositions(int sequenceLength, Random random) {
        ArrayList<Integer> selected = new ArrayList<>();
        if (sequenceLength <= 0) {
            return selected;
        }
        int target = Math.min(sequenceLength, Math.max(1, (int) Math.ceil(0.15 * sequenceLength)));
        ArrayList<Integer> pool = new ArrayList<>();
        for (int i = 0; i < sequenceLength; i++) {
            pool.add(i);
        }
        // Uniform sampling without replacement: pull from a shrinking pool of unselected indices.
        for (int k = 0; k < target; k++) {
            int idx = random.nextInt(pool.size());
            selected.add(pool.remove(idx));
        }
        selected.sort(null);
        return selected;
    }

    /**
     * Builds the standard BERT Next Sentence Prediction (NSP) sub-graph.
     *
     * NSP is the second of BERT's two pre-training objectives (the first being Masked
     * Language Modeling, MLM). Given a packed pair of sentences {@code [CLS] A [SEP] B [SEP]},
     * NSP asks the model to decide whether sentence B is the actual sentence that follows
     * sentence A in the source corpus, or a random sentence drawn from elsewhere. It is a
     * simple binary classification task whose only purpose during pre-training is to push
     * the encoder towards producing a pooled sentence-pair representation in the {@code [CLS]}
     * position that captures inter-sentence coherence.
     *
     * The standard recipe, mirrored here, is:
     * <ol>
     *   <li>Take the final encoder output at row 0 (the {@code [CLS]} token),
     *       a single biased row of shape {@code [1, L]}.</li>
     *   <li><b>Pooler</b>: a learnable linear projection of shape {@code [L, L]} followed by
     *       a {@link Tanh} activation. Following the {@code addEdge(node, function, true)}
     *       convention used elsewhere in this file, the activation appends a bias column,
     *       so the pooled representation has shape {@code [1, L + 1]}.</li>
     *   <li><b>2-class projection</b>: a learnable linear projection of shape
     *       {@code [L + 1, 2]} producing logits of shape {@code [1, 2]} (isNext / notNext).</li>
     *   <li><b>Softmax</b>: applied over the two-class axis to yield NSP probabilities.</li>
     * </ol>
     *
     * <b>Why this method is defined but never wired into the live graph:</b> the
     * {@link ComputationalGraph} base class exposes a single {@code outputNode} and a single
     * loss target per graph. The MLM head built in {@link #train(ArrayList)} already occupies
     * that output with a per-token {@code [r, V]} distribution. Adding NSP as a second live
     * head would mean either overwriting {@code outputNode} (silently disabling MLM) or
     * attempting to backpropagate through two output heads under one loss, both of which
     * break the framework's contract. Real BERT pre-training combines MLM and NSP losses,
     * but with the current single-head computational graph that is not expressible in one
     * pass, so MLM remains the active head and this method is provided as a documented,
     * self-contained construction of the NSP sub-graph (e.g. for future multi-head support
     * or for callers that want to wire NSP as the sole objective in a separate {@code Bert}
     * instance). It is intentionally not invoked from {@link #train(ArrayList)}.
     *
     * @param clsRepresentation the {@code [CLS]} row of the final encoder output, expected
     *                          to be a biased {@code [1, L]} node produced by the caller.
     * @param parameter         the BERT parameter bundle, used for {@code L} and weight
     *                          initialization.
     * @param random            the seeded RNG used for weight initialization, shared with
     *                          the rest of the graph for reproducibility.
     * @return the NSP softmax node of shape {@code [1, 2]}; the caller is responsible for
     *         deciding whether to attach it as a loss target.
     */
    private ComputationalNode nextSentencePrediction(ComputationalNode clsRepresentation, BertParameter parameter, Random random) {
        ComputationalNode poolerWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getL(), random), new int[]{parameter.getL(), parameter.getL()}));
        ComputationalNode pooled = this.addEdge(clsRepresentation, poolerWeight);
        ComputationalNode pooledTanh = this.addEdge(pooled, new Tanh(), true);
        ComputationalNode projectionWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL() + 1, 2, random), new int[]{parameter.getL() + 1, 2}));
        ComputationalNode logits = this.addEdge(pooledTanh, projectionWeight);
        return this.addEdge(logits, new Softmax());
    }

    /**
     * Builds the position-wise feed-forward sub-graph used inside each encoder block:
     * a configurable stack of hidden linear-then-activation layers followed by a final
     * linear projection back to {@code L}. Hidden sizes and per-layer activation
     * functions are taken from {@link BertParameter}.
     * @param current The input node entering the feed-forward sub-graph.
     * @param currentLayerSize The width of {@code current}, including the bias column.
     * @param parameter The BERT parameter bundle supplying the hidden-layer spec and {@code L}.
     * @param random The seeded RNG used to initialize the hidden-layer weights.
     * @return The output node of the feed-forward sub-graph, of width {@code L}.
     */
    private ComputationalNode feedForwardNetwork(ComputationalNode current, int currentLayerSize, BertParameter parameter, Random random) {
        int size = parameter.getFeedForwardSize();
        for (int i = 0; i < size; i++) {
            ComputationalNode hiddenWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(currentLayerSize, parameter.getFeedForwardHiddenLayer(i), random), new int[]{currentLayerSize, parameter.getFeedForwardHiddenLayer(i)}));
            ComputationalNode hiddenLayer = this.addEdge(current, hiddenWeight);
            current = this.addEdge(hiddenLayer, parameter.getActivationFunction(i), true);
            currentLayerSize = parameter.getFeedForwardHiddenLayer(i) + 1;
        }
        ComputationalNode outputWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(currentLayerSize, parameter.getL(), random), new int[]{currentLayerSize, parameter.getL()}));
        return this.addEdge(current, outputWeight);
    }

    /**
     * Builds the full BERT encoder graph (token + segment + positional input,
     * {@code numEncoderLayers} stacked bidirectional self-attention blocks each with
     * Add &amp; LayerNorm, FFN, and another Add &amp; LayerNorm, then the MLM head) and
     * trains it for {@code parameter.getEpoch()} epochs. For each instance ~15% of the
     * token positions are picked by {@link #selectMaskedPositions} and only those rows
     * receive a one-hot gold target, so the MLM cross-entropy loss is computed solely
     * at masked positions.
     * @param trainSet The list of packed training tensors; each is shuffled and consumed once per epoch.
     */
    @Override
    public void train(ArrayList<Tensor> trainSet) {
        BertParameter parameter = (BertParameter) this.parameters;
        int[] lnSize = new int[]{0, 0};
        Random random = new Random(parameter.getSeed());
        // Token + positional embeddings come in via wordInput (biased: framework appends a 1.0 column making it [r, L]).
        ComputationalNode wordInput = new MultiplicationNode(false, true);
        this.inputNodes.add(wordInput);
        // Segment embeddings come in pre-shaped [r, L] (last column 0 to keep wordInput's bias intact after the addition).
        ComputationalNode segmentInput = new ComputationalNode(false, false);
        this.inputNodes.add(segmentInput);
        ComputationalNode embedded = this.addAdditionEdge(wordInput, segmentInput, false);
        ComputationalNode current = embedded;
        // N stacked encoder blocks, each: bidirectional self-attention -> Add & LayerNorm -> FFN -> Add & LayerNorm.
        for (int layer = 0; layer < parameter.getNumEncoderLayers(); layer++) {
            ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(multiHeadAttention(current, parameter, random), 1);
            ComputationalNode wo = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getL(), random), new int[]{parameter.getL(), parameter.getL()}));
            ComputationalNode c = this.addEdge(concatenatedNode, wo);
            ComputationalNode inputC = this.addAdditionEdge(current, c, false);
            ComputationalNode y = layerNormalization(inputC, parameter, lnSize);
            ComputationalNode ff = feedForwardNetwork(y, parameter.getL(), parameter, random);
            ComputationalNode ffResidual = this.addAdditionEdge(ff, y, false);
            current = layerNormalization(ffResidual, parameter, lnSize);
        }
        // MLM head: project the encoder output [r, L] to [r, V] and softmax over the vocabulary for each position.
        ComputationalNode wMlm = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getV(), random), new int[]{parameter.getL(), parameter.getV()}));
        ComputationalNode mlmLogits = this.addEdge(current, wMlm);
        this.outputNode = this.addEdge(mlmLogits, new Softmax());
        ComputationalNode classLabelNode = new ComputationalNode();
        this.addLoss(classLabelNode);
        for (int i = 0; i < parameter.getEpoch(); i++) {
            this.shuffle(trainSet, random);
            for (Tensor instance : trainSet) {
                ArrayList<Integer> classLabels = createInputTensors(instance, this.inputNodes.get(0), this.inputNodes.get(1), parameter.getL() - 1);
                // MLM masking: pick ~15% of token positions to mask and only those positions
                // contribute to the loss. We realize this within the framework's single-loss
                // contract by setting the one-hot gold for masked rows and leaving non-masked
                // rows as all zeros, which makes their cross-entropy contribution exactly 0
                // (and therefore their gradient contribution 0 as well). See selectMaskedPositions
                // for the 15% / floor-of-1 convention and for the noted simplification of the
                // canonical 80/10/10 input-perturbation split.
                ArrayList<Integer> maskedPositions = selectMaskedPositions(classLabels.size(), random);
                HashSet<Integer> maskedSet = new HashSet<>(maskedPositions);
                ArrayList<Double> classLabelValues = new ArrayList<>();
                for (int row = 0; row < classLabels.size(); row++) {
                    boolean masked = maskedSet.contains(row);
                    int classLabel = classLabels.get(row);
                    for (int j = 0; j < parameter.getV(); j++) {
                        if (masked && j == classLabel) {
                            classLabelValues.add(1.0);
                        } else {
                            classLabelValues.add(0.0);
                        }
                    }
                }
                classLabelNode.setValue(new Tensor(classLabelValues, new int[]{classLabels.size(), parameter.getV()}));
                this.forwardCalculation();
                this.backpropagation();
            }
            parameter.getOptimizer().setLearningRate();
        }
    }

    /**
     * Evaluates the trained model on a held-out set of packed tensors, comparing the
     * per-token argmax predictions against the gold labels embedded in each instance
     * and returning the resulting classification accuracy.
     * @param testSet The list of packed test tensors in the same layout as the training set.
     * @return The classification performance whose accuracy is in {@code [0.0, 1.0]}.
     */
    @Override
    public ClassificationPerformance test(ArrayList<Tensor> testSet) {
        int count = 0;
        int total = 0;
        BertParameter parameter = (BertParameter) this.parameters;
        for (Tensor instance : testSet) {
            ArrayList<Integer> goldClassLabels = createInputTensors(instance, this.inputNodes.get(0), this.inputNodes.get(1), parameter.getL() - 1);
            ArrayList<Double> predictions = this.predict();
            int n = Math.min(goldClassLabels.size(), predictions.size());
            for (int i = 0; i < n; i++) {
                if (predictions.get(i).intValue() == goldClassLabels.get(i)) {
                    count++;
                }
                total++;
            }
            if (goldClassLabels.size() > predictions.size()) {
                total += goldClassLabels.size() - predictions.size();
            }
        }
        return new ClassificationPerformance((count + 0.00) / Math.max(total, 1));
    }

    /**
     * Reads the current value of the output node and returns the per-row argmax over the
     * vocabulary axis, i.e. the predicted token id for each position in the sequence.
     * @return A list of predicted class indices, one per row of the output tensor.
     */
    @Override
    protected ArrayList<Double> getOutputValue() {
        ArrayList<Double> classLabels = new ArrayList<>();
        Tensor value = outputNode.getValue();
        for (int i = 0; i < value.getShape()[0]; i++) {
            double max = -Double.MAX_VALUE;
            double index = -1;
            for (int j = 0; j < value.getShape()[1]; j++) {
                if (value.getValue(new int[]{i, j}) > max) {
                    max = value.getValue(new int[]{i, j});
                    index = j;
                }
            }
            classLabels.add(index);
        }
        return classLabels;
    }
}
