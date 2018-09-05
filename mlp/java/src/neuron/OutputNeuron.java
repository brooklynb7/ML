package neuron;

import layer.Layer;

/**
 * OutputNeuron.
 * 
 * @author Daniel Wehner
 *
 */
public class OutputNeuron extends Neuron {
	
	/**
	 * Constructor.
	 * Creates an OutputNeuron.
	 * 
	 * @param position (position of the neuron in the layer)
	 * @param predecCount (number of predecessor neurons in the network)
	 * @param layer (layer the neuron belongs to)
	 */
	public OutputNeuron(int position, int predecCount, Layer layer) {
		super(position, layer);
		this.initializeWeights(predecCount);
	}
	
	/**
	 * Calculates the error which is made during training.
	 * 
	 * @param targets (target value)
	 * @return error value of the neuron
	 */
	@Override
	public double calcError(double[] targets) {
		double o = this.activation;
		this.error = o * (1 - o) * (targets[this.position] - o);
		return this.error;
	}
}
