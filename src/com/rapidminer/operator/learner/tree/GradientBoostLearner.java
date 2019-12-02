
package com.rapidminer.operator.learner.tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.Statistics;
import com.rapidminer.operator.Model;
import com.rapidminer.operator.OperatorCapability;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.learner.AbstractLearner;
import com.rapidminer.operator.learner.functions.XGBRegressionMethod.XGBRegressionResult;
import com.rapidminer.operator.learner.functions.XGBRegressionModel;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

/**
 * <p>
 * This operator calculates a linear regression model. It supports several
 * different mechanisms for model selection: - M5Prime using Akaike criterion
 * for model selection. - A greedy implementation - A T-Test based selection -
 * No selection. Further selections can be added using the static method
 * 
 * </p>
 * 
 * @author Ingo Mierswa
 */
public class GradientBoostLearner extends AbstractLearner {
	
	
	/**
     * The parameter name for &quot;The feature selection method used during
     * regression.&quot;
     */
    public static final String PARAMETER_FEATURE_SELECTION = "feature_selection";

    /**
     * The parameter name for &quot;Indicates if the algorithm should try to
     * delete colinear features during the regression.&quot;
     */
    public static final String PARAMETER_ELIMINATE_COLINEAR_FEATURES = "eliminate_colinear_features";

    public static final String PARAMETER_USE_BIAS = "use_bias";

    /**
     * The parameter name for &quot;The minimum tolerance for the removal of
     * colinear features.&quot;
     */
    public static final String PARAMETER_MIN_TOLERANCE = "min_tolerance";

    /**
     * The parameter name for &quot;The ridge parameter used during ridge
     * regression.&quot;
     */
    public static final String PARAMETER_RIDGE = "ridge";
    
    

	public GradientBoostLearner(OperatorDescription description) {
		super(description);
		// TODO Auto-generated constructor stub
	}

	@Override
	public Model learn(ExampleSet exampleSet) throws OperatorException {
		
		// initializing data and parameter values.
        Attribute label = exampleSet.getAttributes().getLabel();
        Attribute workingLabel = label;
        boolean cleanUpLabel = false;
        String firstClassName = null;
        String secondClassName = null;
        
//        com.rapidminer.example.Tools.onlyNonMissingValues(exampleSet, "Gradient Boost Regression");

//        boolean useBias = getParameterAsBoolean(PARAMETER_USE_BIAS);
//        boolean removeColinearAttributes = getParameterAsBoolean(PARAMETER_ELIMINATE_COLINEAR_FEATURES);
//        double ridge = getParameterAsDouble(PARAMETER_RIDGE);
//        double minTolerance = getParameterAsDouble(PARAMETER_MIN_TOLERANCE);
        
		
        
     // search all attributes and keep numerical
        int numberOfAttributes = exampleSet.getAttributes().size();
        boolean[] isUsedAttribute = new boolean[numberOfAttributes];
        int counter = 0;
        String[] attributeNames = new String[numberOfAttributes];
        for (Attribute attribute : exampleSet.getAttributes()) {
            isUsedAttribute[counter] = attribute.isNumerical();
            attributeNames[counter] = attribute.getName();
            counter++;
        }
        
        
        
        Map<String, Object> params = new HashMap<String, Object>() {
        	  {
        	    put("eta", 1.0);
        	    put("verbosity", 0);
        	    put("max_depth", 2);
        	    put("objective", "reg:squarederror");
        	    put("eval_metric", "rmse");
        	  }
        };
        

        
        
        int nrow = exampleSet.size();
        int ncol = exampleSet.getAttributes().size();

        float[] labels = new float[nrow];

        
        float[] data = new float[nrow*ncol];
        
        int k = 0,j=0;
		for (Example example : exampleSet) {			
		   
			for (Attribute attr: example.getAttributes()) {
				data[k++] = (float) example.getValue(attr);
			}
			//System.out.print("   $" + example.getLabel() + "$   ");
			labels[j++] = (float) example.getLabel();
		}
		
		
		
		
		float missing = 0.0f;
		try {
			final DMatrix trainMat = new DMatrix(data, nrow, ncol, missing);
			final DMatrix testMat = new DMatrix(data, nrow, ncol, missing);
			
			trainMat.setLabel(labels);
			testMat.setLabel(labels);
			

				System.out.println(trainMat.rowNum());
				
			// Specify a watch list to see model accuracy on data sets
			Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
			  {
			    put("train", trainMat);
			    put("test", testMat);
			  }
			};
			int nround = 2;
			Booster booster = XGBoost.train(trainMat, params, nround, watches, null, null);
			System.out.println(booster.getVersion());
		}
		catch(Exception e){
			System.out.println(e);
		}
        
        
       
        // compute and store statistics and turn off attributes with zero
        // standard deviation
        exampleSet.recalculateAllAttributeStatistics();
        double[] means = new double[numberOfAttributes];
        double[] standardDeviations = new double[numberOfAttributes];
        counter = 0;
        Attribute[] allAttributes = new Attribute[exampleSet.getAttributes().size()];
        for (Attribute attribute : exampleSet.getAttributes()) {
            allAttributes[counter] = attribute;
            if (isUsedAttribute[counter]) {
                means[counter] = exampleSet.getStatistics(attribute, Statistics.AVERAGE_WEIGHTED);
                standardDeviations[counter] = Math.sqrt(exampleSet.getStatistics(attribute, Statistics.VARIANCE_WEIGHTED));
                if (standardDeviations[counter] == 0) {
                    isUsedAttribute[counter] = false;
                }
            }
            counter++;
        }
        
        double labelMean = exampleSet.getStatistics(workingLabel, Statistics.AVERAGE_WEIGHTED);
        double labelStandardDeviation = Math.sqrt(exampleSet.getStatistics(workingLabel, Statistics.VARIANCE_WEIGHTED));

        int numberOfExamples = exampleSet.size();

        // determine the number of used attributes + 1
        int numberOfUsedAttributes = 1;
        for (int i = 0; i < isUsedAttribute.length; i++) {
            if (isUsedAttribute[i]) {
                numberOfUsedAttributes++;
            }
        }
        
        XGBRegressionResult result = new XGBRegressionResult();
        result.isUsedAttribute = new boolean[] {true,true};
        double[] standardErrors = {5,6};
        double[] standardizedCoefficients = {3,4};
    	double[] tolerances = {1,2};
		double[] tStatistics = null;
		double[] pValues = null;
		boolean useBias = false;
		//        return new XGBRegressionModel(exampleSet, result.isUsedAttribute, result.coefficients, standardErrors, standardizedCoefficients, tolerances, tStatistics, pValues, useBias, firstClassName, secondClassName);
        return new XGBRegressionModel(exampleSet, result.isUsedAttribute, result.coefficients, standardErrors, standardizedCoefficients, tolerances, tStatistics, pValues, useBias, firstClassName, secondClassName);

	}

    @Override
    public boolean supportsCapability(OperatorCapability capability) {
        if (capability.equals(OperatorCapability.NUMERICAL_ATTRIBUTES))
            return true;
        if (capability.equals(OperatorCapability.NUMERICAL_LABEL))
            return true;
        if (capability.equals(OperatorCapability.BINOMINAL_LABEL))
            return true;
        if (capability == OperatorCapability.WEIGHTED_EXAMPLES)
            return true;
        return false;
    }

   
}
