package layer;

import network.MultiLayerPerceptron;
import neuron.HiddenNeuron;
import neuron.Neuron;
import neuron.OutputNeuron;

/**
 * Layer.
 * 
 * @author Daniel Wehner
 *
 */
public abstract class Layer {
	
	// id of the layer
	private String id;
	
	// size of the layer (number of neurons)
	protected int size;
	
	// array of neurons
	protected Neuron[] neurons;
	
	// next layer in the network
	private Layer next;
	
	// previous layer in the nextwork
	private Layer previous;
	
	// mlp the layer belongs to
	private MultiLayerPerceptron mlp;
	
	// errors of all neurons in this layer
	protected double[] errors;
	
	/**
	 * Constructor.
	 * 
	 * @param id (identifier of the layer)
	 * @param size (size (= number of neurons) of the layer)
	 * @param mlp (MultiLayerPerceptron the layer belongs to)
	 */
	public Layer(String id, int size, MultiLayerPerceptron mlp) {
		this.id = id;
		this.size = size;
		this.mlp = mlp;
		this.neurons = new Neuron[size];
		this.errors = new double[size];
	}
	
	/**
	 * Calculates the activations for all neurons in the layer.
	 * 
	 * @param input (activations from previous layer or input to network)
	 * @return activations for this layer
	 */
	public double[] calcActivations(double[] input) {
		double[] activations = new double[this.size];
		
		for(int n = 0; n < this.neurons.length; n++) {
			Neuron neuron = neurons[n];
			activations[n] = neuron.calcActivation(neuron.calcNet(input));
		}
		return activations;
	}
	
	/**
	 * Calculates the errors made by each neuron
	 * 
	 * @param targets
	 * @return errors
	 */
	public double[] calcErrors(double[] targets) {
		// calculate error for each neuron in the layer
		for(int n = 0; n < this.getSize(); n++) {
			Neuron neuron = this.neurons[n];
			this.errors[n] = neuron.calcError(targets);
		}
		return this.errors;
	}
	
	/**
	 * Updates the weights of all neurons in the layer.
	 * 
	 * @param alpha (learning rate)
	 */
	public void updateWeights(double alpha) {
		for(int n = 0; n < this.size; n++)
			neurons[n].updateWeights(alpha);
	}
	
	/**
	 * Helper method to add neurons to the layer.
	 * 
	 * @param cls (class of the neurons to be created)
	 */
	protected void addNeurons(Class<? extends Neuron> cls) {
		for(int n = 0; n < this.getSize(); n++) {
			// get size of previous layer
			Layer prevLayer = getPrevious();
			int sizePrevLayer = (prevLayer == null)
				? this.getMlp().getDataDimension()
				: prevLayer.getSize();
			
			// create neurons according to class
			this.neurons[n] = (cls == HiddenNeuron.class)
				? new HiddenNeuron(n, sizePrevLayer, this)
				: new OutputNeuron(n, sizePrevLayer, this);
		}
	}
	
	/**
	 * Creates the neurons for this layer.
	 */
	public abstract void createNeurons();
	
	/**
	 * Generates string representation of the layer.
	 */
	@Override
	public String toString() {
		String LF = System.getProperty("line.separator");
		String s = "";
		
		for(int n = 0; n < this.neurons.length; n++)
			s += this.neurons[n].toString() + LF;
		s += "=====" + LF;
		return s;
	}
	
	/*
	 * Getters and Setters 
	 */
	
	public String getId() {
		return this.id;
	}

	public int getSize() {
		return this.size;
	}

	public Neuron[] getNeurons() {
		return this.neurons;
	}

	public Layer getNext() {
		return this.next;
	}

	public void setNext(Layer next) {
		this.next = next;
	}

	public Layer getPrevious() {
		return this.previous;
	}

	public void setPrevious(Layer previous) {
		this.previous = previous;
	}

	public MultiLayerPerceptron getMlp() {
		return this.mlp;
	}

	public double[] getErrors() {
		return this.errors;
	}
}
