package SequenceProcessing.Classification;

import Classification.Parameter.ActivationFunction;
import Classification.Parameter.DeepNetworkParameter;
import Corpus.Sentence;
import SequenceProcessing.Initializer.Initializer;
import SequenceProcessing.Sequence.LabelledVectorizedWord;
import SequenceProcessing.Sequence.SequenceCorpus;
import Math.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class BidirectionalLongShortTermMemory extends LongShortTermMemoryModel implements Serializable {

    private ArrayList<Matrix> fbVectors;
    private ArrayList<Matrix> fbWeights;
    private ArrayList<Matrix> fbRecurrentWeights;
    private ArrayList<Matrix> gbVectors;
    private ArrayList<Matrix> gbWeights;
    private ArrayList<Matrix> gbRecurrentWeights;
    private ArrayList<Matrix> ibVectors;
    private ArrayList<Matrix> ibWeights;
    private ArrayList<Matrix> ibRecurrentWeights;
    private ArrayList<Matrix> obVectors;
    private ArrayList<Matrix> obWeights;
    private ArrayList<Matrix> obRecurrentWeights;
    private ArrayList<Matrix> cbVectors;
    private ArrayList<Matrix> cbOldVectors;
    private ArrayList<Matrix> bLayers;
    private ArrayList<Matrix> bOldLayers;
    private ArrayList<Matrix> bWeights;
    private Matrix outputLayer;


    public void train(SequenceCorpus corpus, DeepNetworkParameter parameters, Initializer initializer) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        ArrayList<Integer> layers = new ArrayList<>();
        layers.add(((LabelledVectorizedWord) corpus.getSentence(0).getWord(0)).getVector().size());
        for (int i = 0; i < parameters.layerSize(); i++) {
            layers.add(parameters.getHiddenNodes(i));
        }
        layers.add(corpus.getClassLabels().size());
        fbVectors = new ArrayList<>();
        fbWeights = new ArrayList<>();
        fbRecurrentWeights = new ArrayList<>();
        gbVectors = new ArrayList<>();
        gbWeights = new ArrayList<>();
        gbRecurrentWeights = new ArrayList<>();
        ibVectors = new ArrayList<>();
        ibWeights = new ArrayList<>();
        ibRecurrentWeights = new ArrayList<>();
        obVectors = new ArrayList<>();
        obWeights = new ArrayList<>();
        obRecurrentWeights = new ArrayList<>();
        cbVectors = new ArrayList<>();
        cbOldVectors = new ArrayList<>();
        bLayers = new ArrayList<>();
        bOldLayers = new ArrayList<>();
        bWeights = new ArrayList<>();
        bLayers.add(new Matrix(layers.get(0), 1));
        outputLayer = new Matrix(corpus.getClassLabels().size(), 1);
        for (int i = 0; i < parameters.layerSize(); i++) {
            fbVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            gbVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            ibVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            obVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            cbVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            cbOldVectors.add(new Matrix(parameters.getHiddenNodes(i), 1));
            bLayers.add(new Matrix(parameters.getHiddenNodes(i), 1));
            bOldLayers.add(new Matrix(parameters.getHiddenNodes(i), 1));
            bWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            fbWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            gbWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            ibWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            obWeights.add(initializer.initialize(layers.get(i + 1), layers.get(i) + 1, new Random(parameters.getSeed())));
            fbRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            gbRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            ibRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
            obRecurrentWeights.add(initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), new Random(parameters.getSeed())));
        }
        bWeights.add(initializer.initialize(layers.get(layers.size() - 1), layers.get(layers.size() - 2) + 1, new Random(parameters.getSeed())));
        bLayers.add(new Matrix(layers.get(layers.size() - 1), 1));
        super.train(corpus, parameters, initializer);
    }

    @Override
    protected void calculateSentence(Sentence sentence, double learningRate) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        for (int k = 0; k < sentence.wordCount(); k++) {
            calculateOutput(sentence, k);
            clear();
        }
        
    }

    @Override
    protected void calculateOutput(Sentence sentence, int index) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        createInputVector(sentence, index);
        ArrayList<Matrix> kVectors = new ArrayList<>();
        ArrayList<Matrix> kbVectors = new ArrayList<>();
        ArrayList<Matrix> jVectors = new ArrayList<>();
        ArrayList<Matrix> jbVectors = new ArrayList<>();
        for (int i = 0; i < this.layers.size() - 2; i++) {
            fVectors.get(i).add(fRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(fWeights.get(i).multiply(this.layers.get(i))));
            fVectors.set(i, activationFunction(fVectors.get(i), this.activationFunction));
            kVectors.add(cOldVectors.get(i).elementProduct(fVectors.get(i)));
            gVectors.get(i).add(gRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(gWeights.get(i).multiply(this.layers.get(i))));
            gVectors.set(i, activationFunction(gVectors.get(i), ActivationFunction.TANH));
            iVectors.get(i).add(iRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(iWeights.get(i).multiply(this.layers.get(i))));
            iVectors.set(i, activationFunction(iVectors.get(i), this.activationFunction));
            jVectors.add(gVectors.get(i).elementProduct(iVectors.get(i)));
            cVectors.get(i).add(jVectors.get(i).sum(kVectors.get(i)));
            oVectors.get(i).add(oRecurrentWeights.get(i).multiply(this.oldLayers.get(i)).sum(oWeights.get(i).multiply(this.layers.get(i))));
            oVectors.set(i, activationFunction(oVectors.get(i), this.activationFunction));
            layers.get(i + 1).add(oVectors.get(i).elementProduct(activationFunction(cVectors.get(i), ActivationFunction.TANH)));
            layers.set(i + 1, biased(layers.get(i + 1)));
            fbVectors.get(i).add(fbRecurrentWeights.get(i).multiply(this.bOldLayers.get(i)).sum(fbWeights.get(i).multiply(this.bLayers.get(i))));
            fbVectors.set(i, activationFunction(fbVectors.get(i), this.activationFunction));
            kbVectors.add(cbOldVectors.get(i).elementProduct(fbVectors.get(i)));
            gbVectors.get(i).add(gbRecurrentWeights.get(i).multiply(this.bOldLayers.get(i)).sum(gbWeights.get(i).multiply(this.bLayers.get(i))));
            gbVectors.set(i, activationFunction(gbVectors.get(i), ActivationFunction.TANH));
            ibVectors.get(i).add(ibRecurrentWeights.get(i).multiply(this.bOldLayers.get(i)).sum(ibWeights.get(i).multiply(this.bLayers.get(i))));
            ibVectors.set(i, activationFunction(ibVectors.get(i), this.activationFunction));
            jbVectors.add(gbVectors.get(i).elementProduct(ibVectors.get(i)));
            cbVectors.get(i).add(jbVectors.get(i).sum(kbVectors.get(i)));
            obVectors.get(i).add(obRecurrentWeights.get(i).multiply(this.bOldLayers.get(i)).sum(obWeights.get(i).multiply(this.bLayers.get(i))));
            obVectors.set(i, activationFunction(obVectors.get(i), this.activationFunction));
            bLayers.get(i + 1).add(obVectors.get(i).elementProduct(activationFunction(cbVectors.get(i), ActivationFunction.TANH)));
            bLayers.set(i + 1, biased(bLayers.get(i + 1)));
        }
        layers.get(layers.size() - 1).add(this.weights.get(this.weights.size() - 1).multiply(layers.get(layers.size() - 2)));
        layers.set(layers.size() - 1, activationFunction(layers.get(layers.size() - 1), this.activationFunction));
        bLayers.get(bLayers.size() - 1).add(this.bWeights.get(this.bWeights.size() - 1).multiply(bLayers.get(bLayers.size() - 2)));
        bLayers.set(bLayers.size() - 1, activationFunction(bLayers.get(bLayers.size() - 1), this.activationFunction));
        normalizeOutput();
    }

    protected Matrix calculateRMinusY(LabelledVectorizedWord word) {
        Matrix r = new Matrix(classLabels.size(), 1);
        int index = classLabels.indexOf(word.getClassLabel());
        r.setValue(index, 0, 1.0);
        for (int i = 0; i < classLabels.size(); i++) {
            r.setValue(i, 0, r.getValue(i, 0) - outputLayer.getValue(i, 0));
        }
        return r;
    }

    protected void createInputVector(Sentence sentence, int index) {
        LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(sentence.wordCount() - 1 - index);
        for (int i = 0; i < bLayers.get(0).getRow(); i++) {
            bLayers.get(0).setValue(i,0, word.getVector().getValue(i));
        }
        bLayers.set(0, biased(bLayers.get(0)));
        super.createInputVector(sentence, index);
    }

    protected void normalizeOutput() {
        for (int i = 0; i < outputLayer.getRow(); i++) {
            outputLayer.setValue(i, 0, layers.get(layers.size() - 1).getValue(i, 0) + bLayers.get(bLayers.size() - 1).getValue(i, 0));
        }
        double sum = 0.0;
        double[] values = new double[outputLayer.getRow()];
        for (int i = 0; i < values.length; i++) {
            sum += Math.exp(outputLayer.getValue(i, 0));
        }
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.exp(outputLayer.getValue(i, 0)) / sum;
        }
        for (int i = 0; i < values.length; i++) {
            outputLayer.setValue(i, 0, values[i]);
        }
    }

    @Override
    protected void oldLayersUpdate() {
        for (int i = 0; i < oldLayers.size(); i++) {
            for (int j = 0; j < oldLayers.get(i).getRow(); j++) {
                bOldLayers.get(i).setValue(j, 0, bLayers.get(i + 1).getValue(j, 0));
                cbOldVectors.get(i).setValue(j, 0, cbVectors.get(i).getValue(j, 0));
                oldLayers.get(i).setValue(j, 0, layers.get(i + 1).getValue(j, 0));
                cOldVectors.get(i).setValue(j, 0, cVectors.get(i).getValue(j, 0));
            }
        }
    }

    @Override
    protected void clear() {
        super.clear();
        for (int l = 0; l < this.layers.size() - 2; l++) {
            for (int m = 0; m < fbVectors.get(l).getRow(); m++) {
                fbVectors.get(l).setValue(m, 0, 0.0);
                gbVectors.get(l).setValue(m, 0, 0.0);
                ibVectors.get(l).setValue(m, 0, 0.0);
                obVectors.get(l).setValue(m, 0, 0.0);
                cbVectors.get(l).setValue(m, 0, 0.0);
            }
        }
    }

    protected void setLayersValuesToZero() {
        for (int j = 0; j < layers.size() - 1; j++) {
            int size = layers.get(j).getRow();
            layers.set(j, new Matrix(size - 1, 1));
            bLayers.set(j, new Matrix(size - 1, 1));
            for (int i = 0; i < layers.get(j).getRow(); i++) {
                layers.get(j).setValue(i, 0, 0.0);
                bLayers.get(j).setValue(i, 0, 0.0);
            }
        }
        for (int i = 0; i < layers.get(layers.size() - 1).getRow(); i++) {
            layers.get(layers.size() - 1).setValue(i, 0, 0.0);
            bLayers.get(bLayers.size() - 1).setValue(i, 0, 0.0);
        }
    }

    @Override
    protected void clearOldValues() {
        for (int i = 0; i < this.oldLayers.size(); i++) {
            for (int k = 0; k < this.oldLayers.get(i).getRow(); k++) {
                cOldVectors.get(i).setValue(k, 0, 0.0);
                this.oldLayers.get(i).setValue(k, 0, 0.0);
                cbOldVectors.get(i).setValue(k, 0, 0.0);
                this.bOldLayers.get(i).setValue(k, 0, 0.0);
            }
        }
    }

    public ArrayList<String> predict(Sentence sentence) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        ArrayList<String> classLabels = new ArrayList<>();
        for (int i = 0; i < sentence.wordCount(); i++) {
            // FIXME: 18.07.2023 todo
            double bestValue = Double.MIN_VALUE;
            String best = this.classLabels.get(0);
            for (int j = 0; j < layers.get(layers.size() - 1).getRow(); j++) {
                if (layers.get(layers.size() - 1).getValue(j, 0) > bestValue) {
                    bestValue = layers.get(layers.size() - 1).getValue(j, 0);
                    best = this.classLabels.get(j);
                }
            }
            classLabels.add(best);
            clear();
        }
        clearOldValues();
        return classLabels;
    }
}
