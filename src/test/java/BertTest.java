import ComputationalGraph.*;
import ComputationalGraph.Function.Sigmoid;
import ComputationalGraph.Function.Tanh;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Optimizer.AdamW;
import Dictionary.VectorizedDictionary;
import Dictionary.Word;
import Dictionary.WordComparator;
import SequenceProcessing.Classification.Bert;
import SequenceProcessing.Parameters.BertParameter;
import org.junit.Test;
import Math.Tensor;

import java.util.ArrayList;
import java.util.Arrays;

public class BertTest {

    @Test
    public void testInitialization() {
        ArrayList<Tensor> tensors = new ArrayList<>();
        tensors.add(new Tensor(Arrays.asList(
                0.2, 0.7, 0.1,
                0.3, 0.4, 0.8,
                0.9, 0.35, 0.12,
                0.27, 0.17, 0.41,
                Double.MAX_VALUE,
                1.0, 6.0, 5.0, 4.0
        ), new int[]{17}));
        tensors.add(new Tensor(Arrays.asList(
                0.2, 0.7, 0.1,
                0.3, 0.4, 0.8,
                0.9, 0.35, 0.12,
                0.27, 0.17, 0.41,
                Double.MAX_VALUE,
                1.0, 6.0, 5.0, 2.0
        ), new int[]{17}));
        tensors.add(new Tensor(Arrays.asList(
                0.2, 0.7, 0.1,
                1.2, 3.6, 7.1,
                5.4, 0.17, 9.8,
                0.77, 0.61, 0.27,
                Double.MAX_VALUE,
                3.0, 4.0, 2.0, 0.0
        ), new int[]{17}));
        ArrayList<Integer> feedForwardHiddenLayers = new ArrayList<>();
        feedForwardHiddenLayers.add(8);
        feedForwardHiddenLayers.add(4);
        ArrayList<Object> activationFunctions = new ArrayList<>();
        activationFunctions.add(new Tanh());
        activationFunctions.add(new Sigmoid());
        ArrayList<Double> gammaValues = new ArrayList<>();
        ArrayList<Double> betaValues = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            gammaValues.add(1.0);
            betaValues.add(0.0);
        }
        ComputationalGraph bert = new Bert(
                new BertParameter(
                        1,
                        1,
                        new AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1),
                        new RandomInitialization(),
                        new CrossEntropyLoss(),
                        3,
                        2,
                        7,
                        2,
                        1e-9,
                        feedForwardHiddenLayers,
                        activationFunctions,
                        gammaValues,
                        betaValues),
                new VectorizedDictionary(new WordComparator() {
                    @Override
                    public int compare(Word word, Word word1) {
                        return 0;
                    }
                }));
        bert.train(tensors);
    }
}
