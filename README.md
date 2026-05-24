For Developers
============
You can also see [C++](https://github.com/StarlangSoftware/SequenceProcessing-CPP), or [C#](https://github.com/StarlangSoftware/SequenceProcessing-CS) repository.
## Requirements

* [Java Development Kit 8 or higher](#java), Open JDK or Oracle JDK
* [Maven](#maven)
* [Git](#git)

### Java 

To check if you have a compatible version of Java installed, use the following command:

    java -version
    
If you don't have a compatible version, you can download either [Oracle JDK](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) or [OpenJDK](https://openjdk.java.net/install/)    

### Maven
To check if you have Maven installed, use the following command:

    mvn --version
    
To install Maven, you can follow the instructions [here](https://maven.apache.org/install.html).      

### Git

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Download Code

In order to work on code, create a fork from GitHub page. 
Use Git for cloning the code to your local or below line for Ubuntu:

	git clone <your-fork-git-link>

A directory called StructureConverter will be created. Or you can use below link for exploring the code:

	git clone https://github.com/starlangsoftware/SequenceProcessing.git

## Open project with IntelliJ IDEA

Steps for opening the cloned project:

* Start IDE
* Select **File | Open** from main menu
* Choose `SequenceProcessing/pom.xml` file
* Select open as project option
* Couple of seconds, dependencies with Maven will be downloaded. 


## Compile

**From IDE**

After being done with the downloading and Maven indexing, select **Build Project** option from **Build** menu. After compilation process, user can run SequenceProcessing.

**From Console**

Go to `SequenceProcessing` directory and compile with 

     mvn compile 

## Generating jar files

**From IDE**

Use `package` of 'Lifecycle' from maven window on the right and from `SequenceProcessing` root module.

**From Console**

Use below line to generate jar file:

     mvn install

## Maven Usage

        <dependency>
            <groupId>io.github.starlangsoftware</groupId>
            <artifactId>SequenceProcessing</artifactId>
            <version>1.0.0</version>
        </dependency>

For Contibutors
============

### pom.xml file
1. Standard setup for packaging is similar to:
```
    <groupId>io.github.starlangsoftware</groupId>
    <artifactId>Amr</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    <name>NlpToolkit.Amr</name>
    <description>Abstract Meaning Representation Library</description>
    <url>https://github.com/StarlangSoftware/Amr</url>

    <organization>
        <name>io.github.starlangsoftware</name>
        <url>https://github.com/starlangsoftware</url>
    </organization>

    <licenses>
        <license>
            <name>The Apache Software License, Version 2.0</name>
            <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
        </license>
    </licenses>

    <developers>
        <developer>
            <name>Olcay Taner Yildiz</name>
            <email>olcay.yildiz@ozyegin.edu.tr</email>
            <organization>Starlang Software</organization>
            <organizationUrl>http://www.starlangyazilim.com</organizationUrl>
        </developer>
    </developers>

    <scm>
        <connection>scm:git:git://github.com/starlangsoftware/amr.git</connection>
        <developerConnection>scm:git:ssh://github.com:starlangsoftware/amr.git</developerConnection>
        <url>http://github.com/starlangsoftware/amr/tree/master</url>
    </scm>

    <properties>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
```
2. Only top level dependencies should be added. Do not forget junit dependency.
```
    <dependencies>
        <dependency>
            <groupId>io.github.starlangsoftware</groupId>
            <artifactId>AnnotatedSentence</artifactId>
            <version>1.0.78</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13.1</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
```
3. Maven compiler, gpg, source, javadoc plugings should be added.
```
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-compiler-plugin</artifactId>
		<version>3.6.1</version>
		<configuration>
			<source>1.8</source>
			<target>1.8</target>
		</configuration>
	</plugin>
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-gpg-plugin</artifactId>
		<version>1.6</version>
		<executions>
			<execution>
				<id>sign-artifacts</id>
				<phase>verify</phase>
				<goals>
					<goal>sign</goal>
				</goals>
			</execution>
		</executions>
	</plugin>
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-source-plugin</artifactId>
		<version>2.2.1</version>
		<executions>
			<execution>
				<id>attach-sources</id>
				<goals>
					<goal>jar-no-fork</goal>
				</goals>
			</execution>
		</executions>
	</plugin>
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-javadoc-plugin</artifactId>
		<configuration>
			<source>8</source>
		</configuration>
		<version>3.10.0</version>
		<executions>
			<execution>
				<id>attach-javadocs</id>
				<goals>
					<goal>jar</goal>
				</goals>
			</execution>
		</executions>
	</plugin>
```
4. Currently publishing plugin is Sonatype.
```
	<plugin>
		<groupId>org.sonatype.central</groupId>
		<artifactId>central-publishing-maven-plugin</artifactId>
		<version>0.8.0</version>
		<extensions>true</extensions>
		<configuration>
			<publishingServerId>central</publishingServerId>
			<autoPublish>true</autoPublish>
		</configuration>
	</plugin>
```
5. For UI jar files use assembly plugins.
```
	<plugin>
		<groupId>org.apache.maven.plugins</groupId>
		<artifactId>maven-assembly-plugin</artifactId>
		<version>2.2-beta-5</version>
		<executions>
			<execution>
				<id>sentence-dependency</id>
				<phase>package</phase>
				<goals>
					<goal>single</goal>
				</goals>
				<configuration>
					<archive>
						<manifest>
							<mainClass>Amr.Annotation.TestAmrFrame</mainClass>
						</manifest>
					</archive>
					<finalName>amr</finalName>
				</configuration>
			</execution>
		</executions>
		<configuration>
			<descriptorRefs>
				<descriptorRef>jar-with-dependencies</descriptorRef>
			</descriptorRefs>
			<appendAssemblyId>false</appendAssemblyId>
		</configuration>
	</plugin>
```
### Resources
1. Add resources to the resources subdirectory. These will include image files (necessary for UI), data files, etc.
   
### Java files
1. Do not forget to comment each function.
```
    /**
     * Returns the value of a given layer.
     * @param viewLayerType Layer for which the value questioned.
     * @return The value of the given layer.
     */
    public String getLayerInfo(ViewLayerType viewLayerType){
```
2. Function names should follow caml case.
```
    public MorphologicalParse getParse()
```
3. Write toString methods, if necessary.
4. Use Junit for writing test classes. Use test setup if necessary.
```
public class AnnotatedSentenceTest {
    AnnotatedSentence sentence0, sentence1, sentence2, sentence3, sentence4;
    AnnotatedSentence sentence5, sentence6, sentence7, sentence8, sentence9;

    @Before
    public void setUp() throws Exception {
        sentence0 = new AnnotatedSentence(new File("sentences/0000.dev"));
```

BERT
============

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only language model that, unlike the encoder–decoder `Transformer` in `SequenceProcessing/Classification/Transformer.java`, drops the autoregressive decoder entirely. Every token in the input is allowed to attend to every other token in both directions at every layer (no causal mask), which makes the resulting representations contextual on both the left and the right. Pre-training is done with two self-supervised objectives: Masked Language Modeling (MLM), where a random subset of input tokens is replaced with `[MASK]` and the model has to predict the original vocabulary id at those positions, and Next Sentence Prediction (NSP), where two sentences are packed as `[CLS] A [SEP] B [SEP]` and the model decides from the pooled `[CLS]` representation whether B is the actual successor of A.

## Architecture mapping to the code

The implementation lives in `src/main/java/SequenceProcessing/Classification/Bert.java`, with hyperparameters in `src/main/java/SequenceProcessing/Parameters/BertParameter.java`. The encoder graph is built end-to-end in `Bert#train`:

* **Input** — token, segment, and positional embeddings are summed before the first encoder block. The token embeddings come in through `wordInput` (a `MultiplicationNode` with the framework's `isBiased=true`, so a `1.0` bias column is appended automatically, giving an `[r, L]` shape). `Bert#positionalEncoding` adds the standard sinusoidal position encoding directly to the token tensor before it is set on `wordInput`. Segment embeddings come in through `segmentInput`, pre-shaped to `[r, L]` with the last column held at `0.0` so the addition into `wordInput` does not disturb the existing bias column. Segment ids are auto-detected by `Bert#matchesSepRow`, which compares each row to the `[SEP]` vector in the supplied `VectorizedDictionary`.
* **N stacked encoder blocks** — each block runs `multiHeadAttention` (N parallel heads computing `softmax(QKᵀ / √dk) · V`, concatenated along axis 1, then projected by an `[L, L]` output weight), followed by **Add & LayerNorm**, then `feedForwardNetwork` (a configurable stack of hidden layers with per-layer activation functions and a final `[currentSize, L]` projection), followed by another **Add & LayerNorm**. `numEncoderLayers` controls how many such blocks are stacked.
* **MLM head** — the encoder output `[r, L]` is projected to `[r, V]` by a learnable `[L, V]` matrix and passed through `Softmax` over the vocabulary axis. This is the head wired into the graph's `outputNode`, and it is the head trained by `forwardCalculation` / `backpropagation`. During training, ~15% of token positions are selected via `Bert#selectMaskedPositions(sequenceLength, random)` and only those *masked* positions contribute to the loss: `train` builds the per-instance gold tensor by setting the one-hot row for masked positions and leaving every non-masked row as all zeros, so cross-entropy on those rows is exactly `−sum(0 · log p) = 0` and they contribute neither value nor gradient. This faithfully reproduces BERT's MLM objective ("predict the masked tokens, not every position") within the framework's single-`outputNode`/single-loss contract without any second head or auxiliary mask tensor.
* **NSP head** — provided as the documented method `Bert#nextSentencePrediction(clsRepresentation, parameter, random)`. It implements the canonical pooler → 2-class projection → Softmax recipe: a learnable `[L, L]` pooler followed by `Tanh`, then a learnable `[L+1, 2]` projection to `isNext` / `notNext` logits, then `Softmax`. The method is intentionally not invoked from `train` (see "Key design decisions" below).

## Key design decisions

* **No attention masking — fully bidirectional.** Unlike the decoder side of `Transformer.java`, which calls `multiHeadAttention(..., isMasked=true)` and inserts a `Mask` node before the softmax, every encoder block in `Bert.java` uses unmasked attention so each position can see the full sequence on both sides.
* **Segment embedding shape `[r, L]` with a zero bias column.** The framework appends a `1.0` bias column to `wordInput`, so the segment input has to be the same width or the `addAdditionEdge` would not line up. `Bert#createInputTensors` builds the segment tensor with `−0.05` (segment 0) or `+0.05` (segment 1) in the first `L − 1` columns and `0.0` in the trailing bias column, so summing `wordInput + segmentInput` leaves the bias column intact for the rest of the graph.
* **Manual LayerNorm with `γ` / `β` pulled per layer.** `Bert#layerNormalization` builds the LN sub-graph (`mean → centered → variance → 1/√(var+ε) → multiply → add`) directly out of `ComputationalGraph` primitives. The `γ` and `β` rows are sourced from `BertParameter` via the `lnSize` counter, so each LN call site consumes its own pair from the supplied arrays. This is intentional: it mirrors the LayerNorm style already used in `Transformer.java` and keeps `BertParameter`'s constructor signature stable.
* **Single-`outputNode` constraint — MLM is the active head, NSP is documented.** The `ComputationalGraph` base class exposes a single `outputNode` and a single loss target per graph. The MLM head already occupies that output with a per-token `[r, V]` distribution, so wiring NSP as a second live head would either silently overwrite `outputNode` or violate the framework's one-head/one-loss contract. NSP is therefore provided as a self-contained method (with full Javadoc) that builds the standard pooler → projection → softmax sub-graph from a `[CLS]` representation but is not called from `train`. MLM stays the active objective.
* **MLM masking (15%).** `Bert#selectMaskedPositions(sequenceLength, random)` picks `ceil(0.15 × sequenceLength)` distinct positions uniformly at random, with a floor of 1 for any non-empty sequence so that the very short sequences used by the unit tests still produce a non-trivial training signal each step. Selection draws from the same `parameter.getSeed()`-seeded `Random` used for weight initialization and shuffling, so masked-position selection is fully reproducible across runs. The canonical BERT 80/10/10 input-perturbation split (replace 80% of selected positions with the `[MASK]` token vector, 10% with a random vocabulary item, 10% unchanged) is left as a noted simplification because applying it cleanly would require a populated `[MASK]` entry in the supplied `VectorizedDictionary` (the existing tests use either an empty dictionary or one without `[MASK]`) plus invasive changes to `createInputTensors` to substitute embedding rows on a per-position basis. Position-level masking on the loss path is the part that materially changes the training objective, and it maps onto the framework's single-loss contract via the zero-target-row trick described in the MLM-head bullet above — no graph changes, no extra loss target.

## Running the BERT tests

The unit tests for the BERT implementation live in `src/test/java/BertTest.java`. They cover (1) end-to-end construction, training and `test()` accuracy reporting on a tiny synthetic dataset with an empty dictionary, (2) the segment-detection path with a populated dictionary that contains a real `[SEP]` entry so `Bert#matchesSepRow` actually fires and the second segment receives the `+0.05` segment embedding, and (3) `testMaskedPositionSelection`, which reflects into the private `Bert#selectMaskedPositions` helper and asserts the masking-policy invariants the architecture relies on: the count rule (`ceil(0.15 × n)` for long sequences, exactly 1 for the short-sequence floor), every returned index is in range `[0, sequenceLength)`, the result is distinct and sorted, the empty-sequence edge case returns an empty list, and a fresh `Random` seeded with the same value reproduces the same masked positions exactly (so training-time masking is deterministic).

From the project root:

    mvn test -Dtest=BertTest

