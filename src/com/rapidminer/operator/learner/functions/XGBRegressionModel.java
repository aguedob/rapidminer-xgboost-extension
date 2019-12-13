
package com.rapidminer.operator.learner.functions;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.Map.Entry;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.UserError;
import com.rapidminer.operator.learner.PredictionModel;
import com.rapidminer.tools.Tools;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;


/**
 * The model for XGBoost regression.
 * 
 * @author Andres Guerrero
 */
public class XGBRegressionModel extends PredictionModel {

	private static final long serialVersionUID = 83812623423432037L;

	private String[] attributeNames;
	
	private String[] attributeConstructions;
	
	private boolean[] selectedAttributes;
		
	private double[] standardErrors;
	
	private double[] tolerances;
	
	private double[] standardizedCoefficients;
	
	private double[] tStatistics;
	
	private double[] pValues;
	
	private boolean useIntercept = true;
	
	private String firstClassName = null;
	
	private String secondClassName = null;
	
	private Booster booster = null;

	private String[] featureNames;

	private int nrounds;

	private String[] evaluationResults;
	
	public XGBRegressionModel(ExampleSet exampleSet,
			boolean[] selectedAttributes, 
			String[] featureNames,
			double[] standardErrors, 
			int nrounds,
			String[] evaluationResults,
			Booster booster) {
		
		super(exampleSet);
		this.attributeNames = com.rapidminer.example.Tools.getRegularAttributeNames(exampleSet);
		this.attributeConstructions = com.rapidminer.example.Tools.getRegularAttributeConstructions(exampleSet);
		this.selectedAttributes = selectedAttributes;
		this.standardErrors = standardErrors;
		this.featureNames = featureNames;
		this.nrounds = nrounds;
		this.booster = booster;
		this.evaluationResults = evaluationResults;
	}
	
	@Override
	public ExampleSet performPrediction(ExampleSet exampleSet, Attribute predictedLabel) throws OperatorException {
		Attribute[] attributes = new Attribute[attributeNames.length];
		for (int i = 0; i < attributeNames.length; i++) {
			attributes[i] = exampleSet.getAttributes().get(attributeNames[i]);
			if (attributes[i] == null && selectedAttributes[i])
				throw new UserError(null, 111, attributeNames[i]);
		}
	
		
		
		int nrow = exampleSet.size();
        int ncol = exampleSet.getAttributes().size();
        float[] data = new float[nrow*ncol];

        int k = 0;
		for (Example example : exampleSet) {			
		   
			for (Attribute attr: example.getAttributes()) {
				
				Double d = example.getValue(attr);
			
				if (d.isNaN()) {
					System.out.println("NAN!" + k);
				}
				data[k++] = (float) example.getValue(attr);
			}

		}

		float missing = 0.0f;
		try {

			final DMatrix predictionMat = new DMatrix(data, nrow, ncol, missing);
			float[][] prediction = this.booster.predict(predictionMat);
			int j=0;
			for (Example example : exampleSet) {			
				   
				double predictionValue = prediction[j++][0];
				example.setValue(predictedLabel, predictionValue );

			}
			
		} catch (XGBoostError e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return exampleSet;
	}
		
	@Override
	public String toString() {
		StringBuffer result = new StringBuffer();
		try {


			if (evaluationResults.length>0) {
				
				result.append("Evaluation Results:\n");
				result.append("========================\n");
				result.append(evaluationResults[evaluationResults.length-1] + "\n\n");
			}
			result.append("Feature Score:\n");
			result.append("========================\n");
			Map<String,Integer> featureScore = booster.getFeatureScore(featureNames);

			Object[] feature = featureScore.entrySet().toArray();
			Arrays.sort(feature, new Comparator() {
				public int compare(Object o1, Object o2) {
					return ((Map.Entry<String, Integer>) o2).getValue()
							   .compareTo(((Map.Entry<String, Integer>) o1).getValue());
				}
			});
			for (Object e : feature) {

				Entry<String, Integer> entry = (Map.Entry<String, Integer>) e;
				
				result.append(entry.getKey() + " : "
				+ entry.getValue() + "\n");
			}

		} 
		catch(Exception e) {
			e.printStackTrace();
		}
		return result.toString();
				
	}
	

	/**
	 * returns an array containing all names of all attributes used for training
	 */
	public String[] getAttributeNames() {
		return attributeNames;
	}
	
	
	public boolean[] getSelectedAttributes() {
		return selectedAttributes;
	}

	
	public double[] getStandardErrors() {
		return standardErrors;
	}


}
