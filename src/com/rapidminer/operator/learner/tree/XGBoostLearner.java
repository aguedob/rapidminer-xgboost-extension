
package com.rapidminer.operator.learner.tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
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
import com.rapidminer.operator.ports.InputPort;
import com.rapidminer.operator.learner.functions.XGBRegressionModel;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeDouble;
import com.rapidminer.parameter.ParameterTypeInt;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

/**
 * <p>
 * This operator calculates a XGBoost regression model.
 * 
 * </p>
 * 
 * @author Andres Guerrerero
 */
public class XGBoostLearner extends AbstractLearner {
	
	
	private final InputPort testSetInput = getInputPorts().createPort("test set");
	

    public static final String PARAMETER_LEARNING_RATE = "learning_rate";
    public static final String PARAMETER_MAX_DEPTH = "max_depth";
    public static final String PARAMETER_N_ROUNDS = "n_rounds";
    public static final String PARAMETER_SUBSAMPLE = "subsample";
    public static final String PARAMETER_ALPHA = "alpha";
    public static final String PARAMETER_LAMBDA = "lambda";
    public static final String PARAMETER_MIN_CHILD_WEIGHT = "min_child_weight";
    public static final String PARAMETER_COL_SAMPLE_BY_TREE= "colsample_by_tree";

    
    

    
    

	public XGBoostLearner(OperatorDescription description) {
		super(description);
	}

	@Override
	public Model learn(ExampleSet exampleSet) throws OperatorException {
		
		
		ExampleSet testSet = testSetInput.getData(ExampleSet.class);
		
        
//        com.rapidminer.example.Tools.onlyNonMissingValues(exampleSet, "Gradient Boost Regression");
   
		
        
        // search all attributes and keep numerical
        int numberOfAttributes = exampleSet.getAttributes().size();
        boolean[] isUsedAttribute = new boolean[numberOfAttributes];
        int counter = 0;
        String[] featureNames = new String[numberOfAttributes];
        for (Attribute attribute : exampleSet.getAttributes()) {
            isUsedAttribute[counter] = attribute.isNumerical();
            featureNames[counter] = attribute.getName();
            counter++;
        }
        
        
        Map<String, Object> params = new HashMap<String, Object>() {


			private static final long serialVersionUID = 1765876987L;

			{
        	    put("eta", getParameterAsDouble(PARAMETER_LEARNING_RATE));
        	    put("verbosity", 1);
        	    put("max_depth", getParameterAsInt(PARAMETER_MAX_DEPTH));
        	    put("objective", "reg:squarederror");
        	    put("eval_metric", "rmse");
        	    put("min_child_weight",getParameterAsDouble(PARAMETER_MIN_CHILD_WEIGHT));
        	    put("alpha",getParameterAsDouble(PARAMETER_ALPHA));
        	    put("lambda",getParameterAsDouble(PARAMETER_LAMBDA));
        	    put("subsample",getParameterAsDouble(PARAMETER_SUBSAMPLE));
    	    	put("colsample_bytree",getParameterAsDouble(PARAMETER_COL_SAMPLE_BY_TREE));
        	    
        	    
        	    
        	}
        };
        

        
        
        int nrow = exampleSet.size();
        int ncol = exampleSet.getAttributes().size();
        int k = 0,j=0;


        float[] labels = new float[nrow];
        float[] data = new float[nrow*ncol];
        
        
		for (Example example : exampleSet) {			
		   
			for (Attribute attr: example.getAttributes()) {
				data[k++] = (float) example.getValue(attr);
			}
			labels[j++] = (float) example.getLabel();

		}
		
		int nrowT = testSet.size();
		int ncolT = testSet.getAttributes().size();
		
		
        float[] labelsT = new float[nrowT];

        
        float[] dataT = new float[nrowT*ncolT];
        
        int kT = 0,jT=0;
		for (Example example : testSet) {			
		   
			for (Attribute attr: example.getAttributes()) {
				dataT[kT++] = (float) example.getValue(attr);
			}
			labelsT[jT++] = (float) example.getLabel();
		}
		
		
		
		Booster booster = null;
		float missing = 0.0f;
		try {
			
			final DMatrix trainMat = new DMatrix(data, nrow, ncol, missing);
			final DMatrix testMat = new DMatrix(dataT, nrowT, ncolT, missing);
			
			trainMat.setLabel(labels);
			testMat.setLabel(labelsT);
							
			// Specify a watch list to see model accuracy on data sets
			Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {

				private static final long serialVersionUID = 198769876L;

			  {
			    put("train", trainMat);
			    put("test", testMat);
			  }
			};
			System.out.println("Training started");
			booster = XGBoost.train(trainMat, params, getParameterAsInt(PARAMETER_N_ROUNDS) , watches, null, null);		
			System.out.println("Training finished");
		}
		catch(Exception e){
			System.out.println("Training failed!");
			e.printStackTrace();
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
        
        
        XGBRegressionResult result = new XGBRegressionResult();
        result.isUsedAttribute = new boolean[] {true,true};
        double[] standardErrors = {5,6};
        return new XGBRegressionModel(exampleSet, 
        		result.isUsedAttribute, 
        		featureNames,
        		standardErrors,
        		getParameterAsInt(PARAMETER_N_ROUNDS),
        		booster);

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
    
    @Override
	public List<ParameterType> getParameterTypes() {
    	List<ParameterType> types = super.getParameterTypes(); 
    	types.add(new ParameterTypeDouble(PARAMETER_LEARNING_RATE, "Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.\n" + 
    			"\n" + 
    			"range: [0,1]" , 0, 1, 0.015, false));
    	types.add(new ParameterTypeInt(PARAMETER_MAX_DEPTH, "Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.\n" + 
    			"\n" + 
    			"range: [0,∞] (0 is only accepted in lossguided growing policy when tree_method is set as hist)" , 0, Integer.MAX_VALUE, 1, false));
    	types.add(new ParameterTypeInt(PARAMETER_N_ROUNDS, "Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.\n" + 
    			"\n" + 
    			"range: [1,∞] The number of rounds for boosting" , 1, Integer.MAX_VALUE, 1, false));
    	types.add(new ParameterTypeDouble(PARAMETER_SUBSAMPLE, "Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.\n" + 
    			"\n" + 
    			"range: [0,∞] The number of rounds for boosting" , 0, Double.MAX_VALUE, 1, false));
    	types.add(new ParameterTypeDouble(PARAMETER_ALPHA, "L1 regularization term on weights. Increasing this value will make model more conservative.\n" 
    			, 0, Double.MAX_VALUE, 0, false));
    	types.add(new ParameterTypeDouble(PARAMETER_LAMBDA, "L2 regularization term on weights. Increasing this value will make model more conservative.\n" 
    			, 0, Double.MAX_VALUE, 1, false));
    	types.add(new ParameterTypeDouble(PARAMETER_MIN_CHILD_WEIGHT, "Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.\n" + 
    			"\n" + 
    			"range: [0,∞] The number of rounds for boosting" , 0, Double.MAX_VALUE, 1, false));
    	types.add(new ParameterTypeDouble(PARAMETER_COL_SAMPLE_BY_TREE, "colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.\n" + 
    			"\n" + 
    			"range: [0,1] The number of features supplied to a tree" , 0, 1, 1, false));
    	return types ;
	}


   
}
