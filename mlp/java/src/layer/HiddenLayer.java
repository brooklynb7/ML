package layer;

import network.MultiLayerPerceptron;
import neuron.HiddenNeuron;

/**
 * HiddenLayer.
 * 
 * @author Daniel Wehner
 *
 */
public class HiddenLayer extends Layer {
	
	/**
	 * Constructor.
	 * Creates a HiddenLayer.
	 * 
	 * @param id (identifier of the layer)
	 * @param size (size (= number of neurons) of the layer)
	 * @param mlp (MultiLayerPerceptron the layer belongs to)
	 */
	public HiddenLayer(String id, int size, MultiLayerPerceptron mlp) {
		super(id, size, mlp);
	}
	
	/**
	 * Creates the neurons for this layer.
	 */
	@Override
	public void createNeurons() {
		this.addNeurons(HiddenNeuron.class);
	}
}
