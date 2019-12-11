
package com.rapidminer.operator.learner.functions;

import java.util.List;

import com.rapidminer.example.ExampleSet;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.UndefinedParameterError;

/**
 * @author Andres Guerrero
 */
public interface XGBRegressionMethod {

	public static class XGBRegressionResult {
		public double[] coefficients;
		public double error;
		public boolean[] isUsedAttribute;
	}
	
	/**
	 * This method performs the actual regression. There are passed the linear regression operator itself as well as data 
	 * and it's properties. Before this method is called, the linear regression already has performed a regression on 
	 * the full data set. This resulted in the given coefficients. Please note, that if useBias is true,
	 * the last coefficient is the bias.
	 * @throws UndefinedParameterError 
	 */
//	public LinearRegressionResult applyMethod(LinearRegression regression, boolean useBias, double ridge, ExampleSet exampleSet, boolean[] isUsedAttribute, int numberOfExamples, int numberOfUsedAttributes, double[] means, double labelMean, double[] standardDeviations, double labelStandardDeviation, double[] coefficientsOnFullData, double errorOnFullData) throws UndefinedParameterError;
	
	/**
	 * This method must return a List of needed Parameters.
	 */
	public List<ParameterType> getParameterTypes();
	
	
	
}
