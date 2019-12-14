
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
	 * This method must return a List of needed Parameters.
	 */
	public List<ParameterType> getParameterTypes();
	
	
	
}
