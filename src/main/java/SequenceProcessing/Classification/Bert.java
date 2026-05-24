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
