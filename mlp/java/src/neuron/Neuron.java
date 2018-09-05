package neuron;

import layer.Layer;

/**
 * Abstract class of a neuron.
 * 
 * @author Daniel Wehner
 *
 */
public abstract class Neuron {	
	
	// id of the neuron
	private String id;
	
	// position of neuron in layer
	protected int position;
	
	// layer the neuron belongs to
	private Layer layer;
	
	// weights of the neuron
	private double[] weights = null;
	
	// inputs that the neuron received
	private double[] inputs = null;
	
	// last activation of the neuron 
	protected double activation;
	
	// last error of the neuron
	protected double error;
	
	/**
	 * Constructor.
	 * 
	 * @param position (position of the neuron in the layer)
	 * @param layer (layer the neuron belongs to)
	 */
	public Neuron(int position, Layer layer) {		
		this.id = layer.getId() + "N" + position;
		this.position = position;
		this.layer = layer;
	}
	
	/**
	 * Initialize all weights to small random numbers.
	 * The number of weights is determined by the number of predecessor neurons in the network.
	 * 
	 * @param predecCount (number of predecessors in the network)
	 */
	protected void initializeWeights(int predecCount) {
		this.weights = new double[predecCount];
		
		for(int i = 0; i < this.weights.length; i++)
			this.weights[i] = Math.random() / 8; // generate random numbers between 0.0 and 0.125
	}
	
	/**
	 * Calculates the net output for that neuron.
	 * Calculation: Sum_i(w_i * x_i) [Multiply each input x_i by its weight w_i and calculate the sum]
	 * 
	 * @param inputs (array of input values)
	 * @return net output of the neuron
	 */
	public double calcNet(double[] inputs) {
		this.inputs = inputs;
		double net = 0;
		
		for(int i = 0; i < inputs.length; i++)
			net += inputs[i] * this.weights[i];
		return net;
	}
	
	/**
	 * Calculates the activation for that neuron.
	 * The activation function is the sigmoid function which takes as argument the net output of the neuron.
	 * Calculation: 1 / (1 + e^(-net))
	 * 
	 * @param net (net output as calculated by calcNetOutput())
	 * @return activation of the neuron
	 */
	public double calcActivation(double net) {
		this.activation = 1 / (1 + Math.pow(Math.E, -net));
		return this.activation;
	}
	
	/**
	 * Updates the weights of the neuron according to the error value.
	 * 
	 * @param alpha (learning rate)
	 */
	public void updateWeights(double alpha) {
		for(int w = 0; w < this.weights.length; w++)
			weights[w] = weights[w] + (alpha * this.error * inputs[w]);
	}
	
	
	/**
	 * Calculates the error which is made during training.
	 * The way the error is calculated varies depending on whether the neuron is a hidden neuron or an output neuron.
	 * 
	 * @param targets (target value = value that should have been predicted)
	 * @return error of the neuron
	 */
	public abstract double calcError(double[] targets);
	
	/**
	 * Generates string representation of the neuron.
	 * 
	 * @return string
	 */
	@Override
	public String toString() {
		String LF = System.getProperty("line.separator");
		String s = "";
		
		for(int w = 0; w < this.weights.length; w++)
			s += this.weights[w] + LF;
		s += "-----";
		return s;
	}
	
	/*
	 * Getters and Setters 
	 */
	
	public String getId() {
		return this.id;
	}
	
	public int getPosition() {
		return this.position;
	}

	public Layer getLayer() {
		return this.layer;
	}

	public double[] getWeights() {
		return this.weights;
	}
	
	public void setWeights(double[] weights) {
		this.weights = weights;
	}

	public double[] getInputs() {
		return this.inputs;
	}

	public double getActivation() {
		return this.activation;
	}

	public double getError() {
		return this.error;
	}
}
