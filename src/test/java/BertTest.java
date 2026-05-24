import Classification.Performance.ClassificationPerformance;
import ComputationalGraph.Function.Sigmoid;
import ComputationalGraph.Function.Tanh;
import ComputationalGraph.Initialization.RandomInitialization;
import ComputationalGraph.Loss.CrossEntropyLoss;
import ComputationalGraph.Optimizer.AdamW;
import Dictionary.VectorizedDictionary;
import Dictionary.VectorizedWord;
import Dictionary.Word;
import Dictionary.WordComparator;
import Math.Tensor;
import Math.Vector;
import SequenceProcessing.Classification.Bert;
import SequenceProcessing.Parameters.BertParameter;
import org.junit.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import static org.junit.Assert.*;

public class BertTest {

    private static VectorizedDictionary emptyDictionary() {
        return new VectorizedDictionary(new WordComparator() {
            @Override
            public int compare(Word word, Word word1) {
                return 0;
            }
        });
    }

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
        Bert bert = new Bert(
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
                emptyDictionary());
        bert.train(tensors);
        ClassificationPerformance performance = bert.test(tensors);
        assertNotNull(performance);
        double accuracy = performance.getAccuracy();
        assertTrue("accuracy should be in [0.0, 1.0] but was " + accuracy, accuracy >= 0.0 && accuracy <= 1.0);
    }

    @Test
    public void testSegmentDetection() {
        // Populate the dictionary with a real [SEP] entry (and a couple of dummy words) so the
        // segment-detection path in Bert#createInputTensors actually fires when a token row
        // matches the [SEP] vector. This is the case the empty-dictionary test never exercises.
        VectorizedDictionary dictionary = new VectorizedDictionary(new WordComparator() {
            @Override
            public int compare(Word word, Word word1) {
                return 0;
            }
        });
        dictionary.addWord(new VectorizedWord("hello", new Vector(new double[]{0.2, 0.7, 0.1})));
        dictionary.addWord(new VectorizedWord("world", new Vector(new double[]{0.3, 0.4, 0.8})));
        dictionary.addWord(new VectorizedWord("[SEP]", new Vector(new double[]{0.5, 0.5, 0.5})));

        // Five token rows × 3 features = 15 doubles, then the Double.MAX_VALUE sentinel,
        // then 5 gold MLM labels (one per row). Row index 2 is the [SEP] row, which matches
        // the dictionary [SEP] vector exactly and flips afterFirstSep so the trailing rows
        // are assigned segment id 1.
        ArrayList<Tensor> tensors = new ArrayList<>();
        tensors.add(new Tensor(Arrays.asList(
                0.2, 0.7, 0.1,
                0.3, 0.4, 0.8,
                0.5, 0.5, 0.5,
                0.9, 0.35, 0.12,
                0.27, 0.17, 0.41,
                Double.MAX_VALUE,
                1.0, 2.0, 3.0, 4.0, 5.0
        ), new int[]{21}));

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
        Bert bert = new Bert(
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
                dictionary);
        bert.train(tensors);
        ClassificationPerformance performance = bert.test(tensors);
        assertNotNull(performance);
        double accuracy = performance.getAccuracy();
        assertTrue("accuracy should be in [0.0, 1.0] but was " + accuracy, accuracy >= 0.0 && accuracy <= 1.0);
    }

    @SuppressWarnings("unchecked")
    private static ArrayList<Integer> invokeSelectMaskedPositions(int sequenceLength, Random random) throws Exception {
        // Reflection: selectMaskedPositions is a private static helper inside Bert (kept private
        // to match the class's other internal helpers like nextSentencePrediction). The test
        // accesses it directly to assert the masking-policy invariants the architecture relies on.
        Method method = Bert.class.getDeclaredMethod("selectMaskedPositions", int.class, Random.class);
        method.setAccessible(true);
        return (ArrayList<Integer>) method.invoke(null, sequenceLength, random);
    }

    @Test
    public void testMaskedPositionSelection() throws Exception {
        // ceil(0.15 * 100) = 15 — for sequences long enough that the 15% rule is the binding
        // constraint, the helper must return exactly that many positions.
        ArrayList<Integer> hundred = invokeSelectMaskedPositions(100, new Random(42));
        assertEquals("expected ceil(0.15 * 100) = 15 masked positions", 15, hundred.size());
        // Distinct, in-range, sorted.
        HashSet<Integer> distinct = new HashSet<>(hundred);
        assertEquals("masked positions must be distinct", hundred.size(), distinct.size());
        for (int idx = 0; idx < hundred.size(); idx++) {
            int position = hundred.get(idx);
            assertTrue("position out of range: " + position, position >= 0 && position < 100);
            if (idx > 0) {
                assertTrue("masked positions should be returned sorted", position > hundred.get(idx - 1));
            }
        }

        // ceil(0.15 * 4) = 1 — for very short sequences the floor-of-1 rule kicks in so each
        // training instance still produces a non-trivial MLM signal.
        ArrayList<Integer> four = invokeSelectMaskedPositions(4, new Random(7));
        assertEquals("short-sequence floor: at least 1 masked position", 1, four.size());
        int single = four.get(0);
        assertTrue("position out of range: " + single, single >= 0 && single < 4);

        // Edge case: empty sequence selects nothing rather than throwing.
        ArrayList<Integer> empty = invokeSelectMaskedPositions(0, new Random(7));
        assertTrue("empty input should select no positions", empty.isEmpty());

        // Determinism: a fresh Random with the same seed must reproduce the same selection
        // exactly, which is what makes training-time masking reproducible across runs.
        ArrayList<Integer> hundredAgain = invokeSelectMaskedPositions(100, new Random(42));
        assertEquals("same seed must produce the same masked positions", hundred, hundredAgain);
    }
}
