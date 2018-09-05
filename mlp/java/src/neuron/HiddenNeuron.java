package neuron;

import layer.Layer;

/**
 * HiddenNeuron.
 * 
 * @author Daniel Wehner
 *
 */
public class HiddenNeuron extends Neuron {
	
	/**
	 * Constructor.
	 * Creates a HiddenNeuron.
	 * 
	 * @param position (position of the neuron in the layer)
	 * @param predecCount (number of predecessor neurons in the network)
	 * @param layer (layer the neuron belongs to)
	 */
	public HiddenNeuron(int position, int predecCount, Layer layer) {
		super(position, layer);
		this.initializeWeights(predecCount);
	}

	/**
	 * Calculates the error which is made during training.
	 * 
	 * @param targets (error values of successor neurons)
	 * @return error value of the neuron
	 */
	@Override
	public double calcError(double[] targets) {
		Neuron[] successors = this.getLayer().getNext().getNeurons();
		double o = this.activation;
		
		for(int n = 0; n < successors.length; n++)
			this.error += targets[n] * successors[n].getWeights()[this.position];
		this.error *= o * (1 - o);
		return this.error;
	}
}
