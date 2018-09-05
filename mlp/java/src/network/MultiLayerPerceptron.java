package network;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;

import layer.HiddenLayer;
import layer.Layer;
import layer.OutputLayer;

/**
 * MultiLayerPerceptron.
 * 
 * @author Daniel Wehner
 *
 */
public class MultiLayerPerceptron {
	
	// array of layer sizes
	// [3, 3, 2] means: two hidden layers with three neurons, one output layer with 2 neurons
	private int[] layerSizes;
	
	// number of layers => layerSizes.length
	private int layerCount;
	
	// dimension of data
	private int dataDimension;
	
	// first layer of the network
	private Layer firstLayer = null;
	
	// last layer of the network
	private OutputLayer outputLayer = null;
	
	// training data (without labels)
	private double[][] predictors;
	
	// training data labels
	private double[][] labels;
	
	/**
	 * Constructor.
	 * Creates a MultiLayerPerceptron.
	 * 
	 * @param hiddenLayerSizes (numbers of neurons per hidden layer)
	 * @param predictors (training data without label)
	 * @param labels (labels for training data)
	 */
	public MultiLayerPerceptron(int[] hiddenLayerSizes, double[][] predictors, double[][] labels) {
		this.layerCount = hiddenLayerSizes.length + 1;
		this.layerSizes = new int[this.layerCount];
		
		for(int i = 0; i < hiddenLayerSizes.length; i++)
			this.layerSizes[i] = hiddenLayerSizes[i];
		this.layerSizes[this.layerCount - 1] = labels[0].length;
		this.predictors = predictors;
		this.labels = labels;
		
		// determine dimensionality of training set
		this.dataDimension = predictors[0].length;
		
		// create all layers and neurons
		this.createLayers();
	}
	
	/**
	 * Constructor.
	 * Creates a MultiLayerPerceptron from a text file.
	 * 
	 * @param filePath (path to text file)
	 */
	public MultiLayerPerceptron(String filePath) {
		this.createFromFile(filePath);
	}
	
	/**
	 * Trains the MultiLayerPerceptron using the training data.
	 * (arrays predictors and labels)
	 * 
	 * @param epochs (number of epochs (= iterations over entire dataset) to perform)
	 * @param alpha (learning rate)
	 * @param cancelIfConverged (stop learning as soon as all examples have been classified correctly)
	 */
	public void train(int epochs, double alpha, boolean cancelIfConverged) {
		System.out.println("MLP: Start training network...");
		double error = 0;
				
		// present training data to the network
		for(int epoch = 0; epoch < epochs; epoch ++) {
			for(int i = 0; i < this.predictors.length; i++) {
				double[] instance = this.predictors[i];
				double[] classLbl = this.labels[i];
				
				this.feedForward(instance); // calculate activations feed-forward
				
				// **************************************************************
				// BACKPROPAGATION
				// **************************************************************
				this.propagateError(classLbl); // propagate the error backwards
				this.updateWeights(alpha);     // update all weights
				
				// calculate classification error
				error += this.outputLayer.calcClassificationError(classLbl);
			}
			
			// cancel training if all training examples are classified correctly
			if(cancelIfConverged)
				if(this.evaluate(this.predictors, this.labels) == 1.0)
					break;
			
			error *= 0.5;
			
			int progress = (epoch * 100) / epochs + 1;
			System.out.println("[" +
					String.join("", Collections.nCopies(progress, "=")) +
					String.join("", Collections.nCopies(100 - progress, " ")) +
			"] Epoch: " + epoch + " Error: " + error);
			error = 0;
		}
		System.out.println("MLP: Network training finished.");
	}
	
	/**
	 * Predicts the classification of the instance.
	 * If the output layer activations as obtained by
	 * @see feedFarward(double[] x) are less than 0.5, 0 is predicted, 1 otherwise. 
	 * 
	 * @param instance
	 * @return prediction
	 */
	public double[] predict(double[] instance) {
		System.out.println("MLP: Predicting instance...");
		double[] prediction = this.feedForward(instance);
		
		// replace outputs < 0.5 by 0, replace by 1 otherwise
		for(int i = 0; i < prediction.length; i++)
			prediction[i] = (prediction[i] >= 0.5) ? 1 : 0;
		return prediction;
	}
	
	/**
	 * Evaluates the accuracy of the trained neural network.
	 * 
	 * @param validationPredictors
	 * @param validationLabels
	 * @return
	 */
	public double evaluate(double[][] validationPredictors, double[][] validationLabels) {
		System.out.println("MLP: Evaluating network...");
		int correct = 0;
		
		for(int i = 0; i < validationPredictors.length; i++) {
			double[] prediction = this.predict(validationPredictors[i]);
			if(Arrays.equals(prediction, validationLabels[i])) correct++;
		}
		return correct / validationPredictors.length;
	}
	
	/**
	 * Exports the neural network to a text file.
	 * 
	 * @param filePath (path of the file)
	 */
	public void saveToFile(String filePath) {
		String LF = System.getProperty("line.separator");
		Layer layer = this.firstLayer;
		
		// create writer
		try(BufferedWriter writer = new BufferedWriter(
			new PrintWriter(filePath)
		)) {
			writer.write("!>>> MLP <<<!" + LF);
			// write data dimension
			writer.write(this.dataDimension + LF);
			// write number of layers
			writer.write(this.layerCount + LF);
			
			// write sizes of layers
			for(int i = 0; i < this.layerSizes.length; i++)
				writer.write(this.layerSizes[i] + LF);
			
			// write weights
			do {
				writer.write(layer.toString());
			} while((layer = layer.getNext()) != null);
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Creates the neural network from the text file specified.
	 * 
	 * @param filePath (path to text file)
	 */
	private void createFromFile(String filePath) {
		try(BufferedReader reader = new BufferedReader(
			new FileReader(filePath)	
		)) {
			if(reader.readLine().equals("!>>> MLP <<<!")) {
				this.dataDimension = Integer.parseInt(reader.readLine()); // read data dimension
				this.layerCount = Integer.parseInt(reader.readLine());    // read number of layers
				this.layerSizes = new int[this.layerCount];			      // create array of layer sizes
				
				for(int i = 0; i < this.layerCount; i++) 				  // fill array layerSizes
					this.layerSizes[i] = Integer.parseInt(reader.readLine());
				this.createLayers();
				
				// read weights and replace random weights by learned weights from file
				Layer layer = this.firstLayer;
				double[] weights = new double[this.dataDimension];
				int i = 0; int n = 0;
				
				String line = reader.readLine();
				do {
					if(line.equals("-----")) {    // new neuron
						line = reader.readLine();
						layer.getNeurons()[n++].setWeights(weights.clone());
						i = 0;
					}
					if(line.equals("=====")) {    // new layer
						line = reader.readLine();
						weights = new double[layer.getSize()];
						layer = layer.getNext();
						n = 0;
					}
					weights[i++] = Double.parseDouble(line);
				} while((line = reader.readLine()) != null);
			}
			else {throw new IllegalArgumentException("Invalid file format");}
		} catch(Exception e) {}
	}
	
	/**
	 * Calculates the output of the MultiLayerPerceptron.
	 * 
	 * @param x (inputs)
	 * @return activations of output layer
	 */
	private double[] feedForward(double[] x) {
		Layer layer = this.firstLayer;
		do {x = layer.calcActivations(x);} while((layer = layer.getNext()) != null);
		
		// variable x now contains the output layer activations
		return x;
	}
	
	/**
	 * Implementation of the backpropagation algorithm.
	 * Propagates the errors from the output layer backwards to the input layer.
	 * 
	 * @param targets
	 */
	private void propagateError(double[] targets) {
		Layer layer = this.outputLayer;
		do {targets = layer.calcErrors(targets);} while((layer = layer.getPrevious()) != null);
	}
	
	/**
	 * Updates all weights in the network.
	 * 
	 * @param alpha (learning rate)
	 */
	private void updateWeights(double alpha) {
		Layer layer = this.firstLayer;
		do {layer.updateWeights(alpha);} while((layer = layer.getNext()) != null);
	}
	
	/**
	 * Creates the layers of the MultiLayerPerceptron.
	 * The layers are implemented as a two-way linked list.
	 */
	private void createLayers() {
		System.out.println("MLP: Creating network...");
		Layer layer = null;
		
		// create 'layerCount' layers
		for(int l = 0; l < this.layerCount; l++) {
			String layerId = "L" + l;
			
			// create first hidden layer
			// *************************
			if(l == 0) {
				layer = new HiddenLayer(layerId, this.layerSizes[l], this);
				layer.createNeurons();
				this.firstLayer = layer;
			}
			
			// create subsequent layers
			// ************************
			else {
				Layer nextLayer = null;
				nextLayer = (l == this.layerCount - 1) 
					? new OutputLayer(layerId, this.layerSizes[l], this)
					: new HiddenLayer(layerId, this.layerSizes[l], this);
				
				// connect layers
				nextLayer.setPrevious(layer);
				nextLayer.createNeurons();
				layer.setNext(nextLayer);
				layer = nextLayer;
			}
		}
		
		// layer now contains the output layer
		this.outputLayer = (OutputLayer) layer;
	}
	
	/*
	 * Getters and Setters
	 */

	public int[] getLayerSizes() {
		return this.layerSizes;
	}

	public int getLayerCount() {
		return this.layerCount;
	}

	public int getDataDimension() {
		return this.dataDimension;
	}

	public Layer getFirstLayer() {
		return this.firstLayer;
	}

	public double[][] getPredictors() {
		return this.predictors;
	}

	public double[][] getLabels() {
		return this.labels;
	}
}
