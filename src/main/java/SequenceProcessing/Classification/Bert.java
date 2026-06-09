package SequenceProcessing.Classification;

import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.*;
import ComputationalGraph.Function.*;
import ComputationalGraph.Node.*;
import Dictionary.*;
import Math.Tensor;
import Math.Vector;
import SequenceProcessing.Functions.FirstRow;
import SequenceProcessing.Functions.Mask;
import SequenceProcessing.Functions.MultiplyByConstant;
import SequenceProcessing.Functions.NspLogitsPadding;
import SequenceProcessing.Functions.PaddingAttentionMask;
import SequenceProcessing.Functions.SelectRow;
import SequenceProcessing.Functions.Transpose;
import SequenceProcessing.Parameters.BertParameter;
import SequenceProcessing.Parameters.TransformerParameter;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

public class Bert extends Transformer {

    private final VectorizedDictionary dictionary;
    private final int sepIndex;
    private int lastMlmRowCount;
    private int lastNspLabel;
    private int maxSequenceLength;
    private final PaddingAttentionMask paddingAttentionMask = new PaddingAttentionMask();

    /**
     * Constructs a BERT model from the given parameter bundle and vectorized dictionary,
     * caching the dictionary index of the {@code [SEP]} token for later segment detection.
     * @param parameter The hyperparameter container; expected to be a {@link BertParameter}.
     * @param dictionary The vectorized dictionary used to look up the {@code [SEP]} entry.
     */
    public Bert(NeuralNetworkParameter parameter, VectorizedDictionary dictionary) {
        super(parameter, dictionary);
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
     * Returns the maximum number of token rows in any packed tensor in {@code tensors},
     * counting only the embedding rows before the {@link Double#MAX_VALUE} sentinel.
     */
    private static int findMaxSequenceLength(ArrayList<Tensor> tensors, int wordEmbeddingLength) {
        int max = 0;
        for (Tensor tensor : tensors) {
            int count = 0;
            for (int i = 0; i < tensor.getShape()[0]; i++) {
                if (tensor.getValue(new int[]{i}) == Double.MAX_VALUE) {
                    break;
                }
                count++;
            }
            int rows = count / wordEmbeddingLength;
            if (rows > max) {
                max = rows;
            }
        }
        return max;
    }

    /**
     * Derives the NSP proxy label from the per-row segment ids assigned during parsing.
     * Returns {@code 1} when at least one token row was assigned segment id {@code 1}
     * (content after the first {@code [SEP]}), and {@code 0} otherwise.
     *
     * This is a <b>structural proxy</b>, not gold next-sentence supervision: the packed
     * training tensors carry no {@code isNext} bit and no negative-sentence sampling.
     * @param segmentIds The segment id recorded for each token row before tensors are built.
     * @return {@code 1} for a detected two-segment instance, {@code 0} for single-segment.
     */
    private static int deriveNspLabel(ArrayList<Integer> segmentIds) {
        for (int sid : segmentIds) {
            if (sid == 1) {
                return 1;
            }
        }
        return 0;
    }

    /**
     * Parses a packed training instance into the BERT word and segment input tensors and
     * returns the gold token ids plus the derived NSP proxy label. Token embeddings populate
     * {@code wordInput} (with positional encoding applied), while {@code segmentInput}
     * receives the segment-id tensor whose ids flip at the first {@code [SEP]} row and whose
     * trailing bias column is held at {@code 0.0}.
     * @param instance The packed input tensor: embedding rows, {@code Double.MAX_VALUE}, then class labels.
     * @param wordInput The graph input node that receives the token embedding tensor.
     * @param segmentInput The graph input node that receives the segment embedding tensor.
     * @param wordEmbeddingLength The width of a single token embedding row.
     * @return The list of gold class labels (one per token row) read from the instance.
     *         The structural NSP proxy label for the same instance is stored in
     *         {@link #lastNspLabel} and may be read immediately after this call.
     */
    @Override
    protected ArrayList<Integer> createInputTensors(Tensor instance, ComputationalNode wordInput, ComputationalNode segmentInput, int wordEmbeddingLength) {
        boolean isOutput = false;
        ArrayList<Integer> classLabels = new ArrayList<>();
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Integer> segmentIds = new ArrayList<>();
        boolean afterFirstSep = false;
        int nspLabel = 0;
        int L = wordEmbeddingLength + 1;
        for (int i = 0; i < instance.getShape()[0]; i++) {
            double val = instance.getValue(new int[]{i});
            if (val == Double.MAX_VALUE) {
                isOutput = true;
                int rows = values.size() / wordEmbeddingLength;
                nspLabel = deriveNspLabel(segmentIds);
                while (values.size() < maxSequenceLength * wordEmbeddingLength) {
                    for (int p = 0; p < wordEmbeddingLength; p++) {
                        values.add(0.0);
                    }
                    segmentIds.add(0);
                }
                Tensor wordTensor = new Tensor(values, new int[]{maxSequenceLength, wordEmbeddingLength});
                wordInput.setValue(positionalEncoding(wordTensor, wordEmbeddingLength));
                ArrayList<Double> segFlat = new ArrayList<>();
                for (int r = 0; r < maxSequenceLength; r++) {
                    int sid = r < segmentIds.size() ? segmentIds.get(r) : 0;
                    double segValue = (sid == 0) ? -0.05 : 0.05;
                    for (int c = 0; c < wordEmbeddingLength; c++) {
                        segFlat.add(segValue);
                    }
                    segFlat.add(0.0);
                }
                segmentInput.setValue(new Tensor(segFlat, new int[]{maxSequenceLength, L}));
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
        lastMlmRowCount = classLabels.size();
        lastNspLabel = nspLabel;
        paddingAttentionMask.setValidLength(lastMlmRowCount);
        return classLabels;
    }

    /**
     * Bidirectional multi-head self-attention with per-instance padding mask on the
     * pre-softmax scores. Padded rows/columns (index {@code >=} the real token count
     * tracked in {@link #lastMlmRowCount}) receive additive {@code -30.0} so they do
     * not attend to or receive attention from real tokens. When every row is real
     * ({@code validLength == maxSequenceLength}), the map is identical to the parent
     * unmasked path.
     */
    @Override
    protected ArrayList<ComputationalNode> multiHeadAttention(ComputationalNode input, TransformerParameter parameter, boolean isMasked, Random random) {
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
            ComputationalNode sQkDk;
            if (isMasked) {
                ComputationalNode mQkDk = this.addEdge(qkDk, new Mask());
                sQkDk = this.addEdge(mQkDk, new Softmax());
            } else {
                ComputationalNode paddedQkDk = this.addEdge(qkDk, paddingAttentionMask);
                sQkDk = this.addEdge(paddedQkDk, new Softmax());
            }
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
     * remaining 85% of positions are ignored by the loss. Position-level masking on
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
     * Builds the pre-softmax BERT Next Sentence Prediction (NSP) sub-graph from a
     * {@code [CLS]} row of the final encoder output.
     *
     * The sub-graph is: pooler {@code [L, L]} → {@link Tanh} (with bias) → projection
     * {@code [L + 1, 2]} producing logits of shape {@code [1, 2]} (isNext / notNext).
     * No {@link Softmax} is applied here; the caller pads and concatenates these logits
     * with the MLM head under a single row-wise {@link Softmax} in {@link #train(ArrayList)}.
     *
     * @param clsRepresentation the {@code [CLS]} row of the final encoder output, expected
     *                          to be a biased {@code [1, L]} node produced by the caller.
     * @param parameter         the BERT parameter bundle, used for {@code L} and weight
     *                          initialization.
     * @param random            the seeded RNG used for weight initialization, shared with
     *                          the rest of the graph for reproducibility.
     * @return the NSP logits node of shape {@code [1, 2]} (pre-softmax).
     */
    private ComputationalNode nextSentencePredictionLogits(ComputationalNode clsRepresentation, BertParameter parameter, Random random) {
        ComputationalNode poolerWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getL(), random), new int[]{parameter.getL(), parameter.getL()}));
        ComputationalNode pooled = this.addEdge(clsRepresentation, poolerWeight);
        ComputationalNode pooledTanh = this.addEdge(pooled, new Tanh(), true);
        ComputationalNode projectionWeight = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL() + 1, 2, random), new int[]{parameter.getL() + 1, 2}));
        return this.addEdge(pooledTanh, projectionWeight);
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
     * Add &amp; LayerNorm, FFN, and another Add &amp; LayerNorm) and trains it for
     * {@code parameter.getEpoch()} epochs with <b>joint MLM + NSP</b> under the framework's
     * single-{@code outputNode}/single-loss contract.
     *
     * <p>MLM: ~15% of token positions are masked via {@link #selectMaskedPositions}; only
     * masked rows receive a one-hot gold target. NSP: the final row of the joint
     * {@code [r+1, V]} output is the padded NSP head; its gold target is a 2-class one-hot
     * in columns {@code 0} (isNext) and {@code 1} (notNext). The NSP label is a
     * <b>structural proxy</b> from segment detection (segment id {@code 1} present or not),
     * not corpus-level next-sentence supervision.</p>
     *
     * <p>Graph wiring: MLM logits {@code [r, V]} and NSP logits padded to {@code [1, V]}
     * are concatenated along dim {@code 0}, then a single row-wise {@link Softmax} forms
     * {@code outputNode}.</p>
     *
     * @param trainSet The list of packed training tensors; each is shuffled and consumed once per epoch.
     */
    @Override
    public void train(ArrayList<Tensor> trainSet) {
        BertParameter parameter = (BertParameter) this.parameters;
        int wordEmbeddingLength = parameter.getL() - 1;
        maxSequenceLength = findMaxSequenceLength(trainSet, wordEmbeddingLength);
        int[] lnSize = new int[4];
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
            ConcatenatedNode concatenatedNode = (ConcatenatedNode) this.concatEdges(multiHeadAttention(current, parameter, false, random), 1);
            ComputationalNode wo = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getL(), random), new int[]{parameter.getL(), parameter.getL()}));
            ComputationalNode c = this.addEdge(concatenatedNode, wo);
            ComputationalNode inputC = this.addAdditionEdge(current, c, false);
            ComputationalNode y = layerNormalization(inputC, parameter, true, lnSize);
            ComputationalNode ff = feedForwardNetwork(y, parameter.getL(), parameter, random);
            ComputationalNode ffResidual = this.addAdditionEdge(ff, y, false);
            current = layerNormalization(ffResidual, parameter, true, lnSize);
        }
        // Joint MLM + NSP head under a single Softmax (single-output / single-loss contract).
        // Each MLM row and the NSP row are separate [1, V] parents so concat backward splits
        // evenly (the framework assumes equal blocks along the concat dimension).
        ComputationalNode wMlm = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getV(), random), new int[]{parameter.getL(), parameter.getV()}));
        ArrayList<ComputationalNode> jointParents = new ArrayList<>();
        for (int k = 0; k < maxSequenceLength; k++) {
            ComputationalNode rowK = this.addEdge(current, new SelectRow(k));
            jointParents.add(this.addEdge(rowK, wMlm));
        }
        ComputationalNode clsRow = this.addEdge(current, new FirstRow());
        ComputationalNode nspLogits = nextSentencePredictionLogits(clsRow, parameter, random);
        ComputationalNode nspPadded = this.addEdge(nspLogits, new NspLogitsPadding(parameter.getV()));
        jointParents.add(nspPadded);
        ComputationalNode jointLogits = this.concatEdges(jointParents, 0);
        this.outputNode = this.addEdge(jointLogits, new Softmax());
        ComputationalNode classLabelNode = new ComputationalNode();
        this.addLoss(classLabelNode);
        for (int i = 0; i < parameter.getEpoch(); i++) {
            this.shuffle(trainSet, random);
            for (Tensor instance : trainSet) {
                ArrayList<Integer> classLabels = createInputTensors(instance, this.inputNodes.get(0), this.inputNodes.get(1), parameter.getL() - 1);
                int nspLabel = lastNspLabel;
                // MLM masking: pick ~15% of token positions to mask and only those positions
                // contribute to the loss. We realize this within the framework's single-loss
                // contract by setting the one-hot gold for masked rows and leaving non-masked
                // rows as all zeros, which makes their cross-entropy contribution exactly 0
                // (and therefore their gradient contribution 0 as well). See selectMaskedPositions
                // for the 15% / floor-of-1 convention.
                ArrayList<Integer> maskedPositions = selectMaskedPositions(classLabels.size(), random);
                HashSet<Integer> maskedSet = new HashSet<>(maskedPositions);
                ArrayList<Double> classLabelValues = new ArrayList<>();
                for (int row = 0; row < maxSequenceLength; row++) {
                    if (row < classLabels.size()) {
                        boolean masked = maskedSet.contains(row);
                        int classLabel = classLabels.get(row);
                        for (int j = 0; j < parameter.getV(); j++) {
                            if (masked && j == classLabel) {
                                classLabelValues.add(1.0);
                            } else {
                                classLabelValues.add(0.0);
                            }
                        }
                    } else {
                        for (int j = 0; j < parameter.getV(); j++) {
                            classLabelValues.add(0.0);
                        }
                    }
                }
                // NSP gold row (structural proxy: segment id 1 present => isNext in column 0).
                // The packed tensors carry no true isNext bit and no negative-sentence sampling.
                for (int j = 0; j < parameter.getV(); j++) {
                    if (j == 0 && nspLabel == 1) {
                        classLabelValues.add(1.0);
                    } else if (j == 1 && nspLabel == 0) {
                        classLabelValues.add(1.0);
                    } else {
                        classLabelValues.add(0.0);
                    }
                }
                classLabelNode.setValue(new Tensor(classLabelValues, new int[]{maxSequenceLength + 1, parameter.getV()}));
                this.forwardCalculation();
                this.backpropagation();
            }
            parameter.getOptimizer().setLearningRate();
        }
    }

    /**
     * Evaluates the trained model on a held-out set of packed tensors, comparing the
     * per-token argmax predictions against the gold labels embedded in each instance
     * and returning the resulting classification accuracy. The extra NSP output row is
     * excluded from MLM scoring.
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
     * vocabulary axis for the MLM token rows only (rows {@code 0 .. r-1}), excluding the
     * trailing NSP row at index {@code r}.
     * @return A list of predicted class indices, one per MLM token row.
     */
    @Override
    protected ArrayList<Double> getOutputValue() {
        ArrayList<Double> classLabels = new ArrayList<>();
        Tensor value = outputNode.getValue();
        int mlmRows = lastMlmRowCount;
        for (int i = 0; i < mlmRows; i++) {
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
