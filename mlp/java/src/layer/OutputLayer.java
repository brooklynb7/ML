package layer;

import network.MultiLayerPerceptron;
import neuron.OutputNeuron;

/**
 * OutputLayer.
 * 
 * @author Daniel Wehner
 *
 */
public class OutputLayer extends Layer {

	/**
	 * Constructor.
	 * Creates an OutputLayer.
	 * 
	 * @param id (identifier of the layer)
	 * @param size (size (= number of neurons) of the layer)
	 * @param mlp (MultiLayerPerceptron the layer belongs to)
	 */
	public OutputLayer(String id, int size, MultiLayerPerceptron mlp) {
		super(id, size, mlp);
	}
	
	/**
	 * Creates the neurons for this layer.
	 */
	@Override
	public void createNeurons() {
		this.addNeurons(OutputNeuron.class);
	}
	
	/**
	 * Calculates the classification error made in the
	 * OutputLayer.
	 * 
	 * @param targets
	 * @return classification error
	 */
	public double calcClassificationError(double[] targets) {
		double error = 0;
		
		for(int n = 0; n < this.size; n++)
			error += Math.pow((targets[n] - this.neurons[n].getActivation()), 2);
		return error;
	}
}
