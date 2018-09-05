package dataprovider;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * DataProvider. Loads data from files, provides shuffling and data splitting
 * (training and test)
 * 
 * @author Daniel Wehner
 *
 */
public class DataProvider {
	private String filePathData;
	private String filePathLabels;
	private double trainRatio;

	private double[][] data;
	private double[][] labels;
	private double[][] trainData;
	private double[][] testData;
	private double[][] trainLabels;
	private double[][] testLabels;

	private Random random;

	/**
	 * Constructor. Reads data from file.
	 * 
	 * @param filePathData   (path to file to read data from)
	 * @param filePathLabels (path to file to read labels from)
	 * @param trainRatio     (fraction of data to be used for training the remaining
	 *                       fraction will be used for testing)
	 * @param shuffle        (flag that indicates if shuffling is activated or
	 *                       deactivated)
	 */
	public DataProvider(String filePathData, String filePathLabels, double trainRatio, boolean shuffle) {
		this.filePathData = filePathData;
		this.filePathLabels = filePathLabels;
		this.trainRatio = trainRatio;

		System.out.println("DataProvider: Reading data...");

		// read data and labels
		this.data = readDataFromFile(filePathData);
		this.labels = readDataFromFile(filePathLabels);

		if (shuffle) {
			System.out.println("DataProvider: Shuffling data...");

			// create random number generator
			this.random = new Random(System.currentTimeMillis());

			// shuffle data and labels likewise (same random seed)
			shuffle(this.data);
			shuffle(this.labels);
		}

		System.out.println("DataProvider: Splitting data...");
		// split the data
		split(trainRatio);

		System.out.println("DataProvider: Finished preparing data.");
	}

	/**
	 * Reads the data from the file specified
	 * 
	 * @param filePath (path to file to be read)
	 * @return
	 */
	private double[][] readDataFromFile(String filePath) {
		double[][] result = null;

		try {
			// read all lines to a list
			List<String> lines = Files.readAllLines(Paths.get(filePath), Charset.defaultCharset());

			result = new double[lines.size()][];

			for (int i = 0; i < lines.size(); i++) {
				result[i] = convertToDouble(lines.get(i).split(","));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return result;
	}

	/**
	 * Converts string array to double array.
	 * 
	 * @param arr (string array to be converted)
	 * @return (double array)
	 */
	private double[] convertToDouble(String[] arr) {
		double[] result = new double[arr.length];

		for (int i = 0; i < arr.length; i++) {
			result[i] = Double.parseDouble(arr[i]);
		}

		return result;
	}

	/**
	 * Shuffles data randomly.
	 * 
	 * @param arr (array to be shuffled)
	 */
	private void shuffle(double[][] arr) {
		for (int i = arr.length - 1; i > 0; i--) {
			int index = this.random.nextInt(i + 1);

			// simple swap
			double[] item = arr[index];
			arr[index] = arr[i];
			arr[i] = item;
		}
	}

	/**
	 * Splits data into train set and test set
	 * 
	 * @param trainRatio (fraction of data to be used for training the remaining
	 *                   fraction will be used for testing)
	 */
	private void split(double trainRatio) {
		int len = this.data.length;
		int splitIndex = (int) Math.ceil(len * trainRatio);

		// split data into train set and test set
		this.trainData = Arrays.copyOfRange(this.data, 0, splitIndex);
		this.testData = Arrays.copyOfRange(this.data, splitIndex, len);

		this.trainLabels = Arrays.copyOfRange(this.labels, 0, splitIndex);
		this.testLabels = Arrays.copyOfRange(this.labels, splitIndex, len);
	}

	/*
	 * Getters and Setters
	 */

	public double[][] getTrainData() {
		return this.trainData;
	}

	public double[][] getTrainLabels() {
		return this.trainLabels;
	}

	public double[][] getTestData() {
		return this.testData;
	}

	public double[][] getTestLabels() {
		return this.testLabels;
	}

	public String getFilePathData() {
		return filePathData;
	}

	public String getFilePathLabels() {
		return filePathLabels;
	}

	public double getTrainRatio() {
		return trainRatio;
	}
}
