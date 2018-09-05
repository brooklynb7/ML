package main;

import dataprovider.DataProvider;
import network.MultiLayerPerceptron;

/**
 * Main.
 * 
 * @author Daniel Wehner
 *
 */
@SuppressWarnings("all")
public class Main {
	private final static int EPOCHS = 250000; // number of iterations over training data
	private final static double ALPHA = 0.05; // learning rate

	/**
	 * Main method. Gets the data, trains a network and makes predictions.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		// GET DATA AND LABELS
		// ***************************************************************
		DataProvider provider = new DataProvider(
				"/Users/brooklynb7/OneDrive/Brooklynb7/MachineLearning/SAP/Common ML Basics 3/mlp/mlp/data/addition/data.txt",
				"/Users/brooklynb7/OneDrive/Brooklynb7/MachineLearning/SAP/Common ML Basics 3/mlp/mlp/data/addition/labels.txt",
				1.0, false);

		double[][] trainData = provider.getTrainData();
		double[][] trainLabels = provider.getTrainLabels();

		double[][] testData = provider.getTestData();
		double[][] testLabels = provider.getTestLabels();

		// TRAIN THE NETWORK
		// ***************************************************************
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[] { 25, 20, 15, 10 }, trainData, trainLabels);

		mlp.train(EPOCHS, ALPHA, false);

		// or alternatively: read mlp from file
		// MultiLayerPerceptron mlp = new MultiLayerPerceptron(
		// "/Users/Daniel/Documents/r/ki/mlp/saved/mlp.txt"
		// );

		// save trained model
		// mlp.saveToFile("/Users/D062271/Documents/KI_DM_ML/algorithms/mlp/saved/mlp_cpu_timeseries.txt");

		// EVALUATE THE NETWORK
		// ***************************************************************
		// System.out.println(mlp.evaluate(testData, testLabels));

		// MAKE PREDICTIONS
		// ***************************************************************
		double[] prediction = mlp.predict(new double[] { 3, 3 });

		System.out.println("Prediction:");
		for (int j = 0; j < prediction.length; j++)
			System.out.println(prediction[j]);
	}
}
