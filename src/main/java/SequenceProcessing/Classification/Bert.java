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

import java.util.ArrayList;
import java.util.Random;

public class Bert extends ComputationalGraph {

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
        int curLength = 0;
        ArrayList<Integer> classLabels = new ArrayList<>();
        ArrayList<Double> values = new ArrayList<>();
        ArrayList<Integer> segmentIds = new ArrayList<>();
        boolean afterFirstSep = false;
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
                    segFlat.add(sid == 0 ? 1.0 : 0.0);
                    segFlat.add(sid == 0 ? 0.0 : 1.0);
                }
                segmentInput.setValue(new Tensor(segFlat, new int[]{rows, 2}));
                curLength = 0;
                values.clear();
                segmentIds.clear();
                afterFirstSep = false;
            } else if (isOutput) {
                if ((curLength + 1) % (wordEmbeddingLength + 1) == 0) {
                    classLabels.add((int) val);
                } else {
                    values.add(val);
                }
                curLength++;
            } else {
                values.add(val);
                curLength++;
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
        ComputationalNode inputC1Mean = this.addEdge(input, new Mean());
        ComputationalNode mean1Minus = this.addEdge(inputC1Mean, new Negation());
        ComputationalNode inputC1Mean1Minus = this.addAdditionEdge(input, mean1Minus, false);
        ComputationalNode variance1 = this.addEdge(inputC1Mean1Minus, new Variance());
        ComputationalNode rootVariance1 = this.addEdge(variance1, new SquareRoot(parameter.getEpsilon()));
        ComputationalNode inverseRootVariance1 = this.addEdge(rootVariance1, new Inverse());
        ComputationalNode lnValue1 = this.addEdge(inputC1Mean1Minus, inverseRootVariance1, false, true);
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getGammaValue(lnSize[0]));
        }
        lnSize[0]++;
        ComputationalNode gammaNode = new MultiplicationNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}), true);
        ComputationalNode lnValue1Gamma = this.addEdge(lnValue1, gammaNode);
        data.clear();
        for (int j = 0; j < parameter.getL(); j++) {
            data.add(parameter.getBetaValue(lnSize[1]));
        }
        lnSize[1]++;
        ComputationalNode betaNode = new ComputationalNode(true, false, new Tensor(data, new int[]{1, parameter.getL()}));
        return this.addAdditionEdge(lnValue1Gamma, betaNode, false);
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
        ComputationalNode wordInput = new MultiplicationNode(false, true);
        this.inputNodes.add(wordInput);
        ComputationalNode segmentInput = new MultiplicationNode(false, true);
        this.inputNodes.add(segmentInput);
        ComputationalNode wSeg = new MultiplicationNode(new Tensor(parameter.initializeWeights(2, parameter.getL() - 1, random), new int[]{2, parameter.getL() - 1}));
        ComputationalNode segProj = this.addEdge(segmentInput, wSeg);
        ComputationalNode embedded = this.addAdditionEdge(wordInput, segProj, false);
        ComputationalNode current = embedded;
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
        ComputationalNode wMlm = new MultiplicationNode(new Tensor(parameter.initializeWeights(parameter.getL(), parameter.getV(), random), new int[]{parameter.getL(), parameter.getV()}));
        ComputationalNode mlmLogits = this.addEdge(current, wMlm);
        this.outputNode = this.addEdge(mlmLogits, new Softmax());
        ComputationalNode classLabelNode = new ComputationalNode();
        this.addLoss(classLabelNode);
        for (int i = 0; i < parameter.getEpoch(); i++) {
            this.shuffle(trainSet, random);
            for (Tensor instance : trainSet) {
                ArrayList<Integer> classLabels = createInputTensors(instance, this.inputNodes.get(0), this.inputNodes.get(1), parameter.getL() - 1);
                ArrayList<Double> classLabelValues = new ArrayList<>();
                for (Integer classLabel : classLabels) {
                    for (int j = 0; j < parameter.getV(); j++) {
                        if (j == classLabel) {
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
            this.forwardCalculation();
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
            double max = Double.MIN_VALUE;
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
