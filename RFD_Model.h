
#ifndef RFD_Model_H
#define RFD_Model_H

#include <string.h>
#include <stdio.h>

#include "FloatMat.h"
#include "DecisionTree.h"

//
class RFD_Model
{
private:
	int NumTrees;      //
	//
	int NumFeatures;
	int NumTypes;

public:
	DecisionTree * ArrayTrees;
	int * ArrayNumNodes;
	//
	// Paras for random
	float FeaturePortion;      //
	float SamplePortion;      //
	//
	// Paras for single tree grow
	static ParasBDT paras;
	//
	// 不剪枝，特征重用
	//

	//
	RFD_Model(void)
	{
		// forest size
		NumTrees = 0;
		ArrayTrees = NULL;
		ArrayNumNodes = NULL;
		//
		NumFeatures = 0;
		NumTypes = 0;
		//
		// Paras for random
		FeaturePortion = 0.5;
		SamplePortion = 0.5;
		//
	} // construct
	~RFD_Model()
	{
		if (ArrayTrees != NULL) delete [] ArrayTrees;
		if (ArrayNumNodes != NULL) delete [] ArrayNumNodes;
	}
	//
	void setNumTrees(int Num)
	{
		NumTrees = Num;
		ArrayTrees = new DecisionTree[Num];
		ArrayNumNodes = new int[Num];
	}
	void setNumInputOutput(int Num1, int Num2)
	{
		NumFeatures = Num1;
		NumTypes = Num2;
	}
	int getNumTrees()
	{
		return NumTrees;
	}
	int getNumFeatures()
	{
		return NumFeatures;
	}
	int getNumTypes()
	{
		return NumTypes;
	}
	//
	void display()
	{
		printf("A RandomForestDecision Model with:\n");
		//
		printf("NumTrees: %d\n", NumTrees);
		printf("NumFeatures: %d\n", NumFeatures);
		printf("NumTypes: %d\n", NumTypes);
		//
		printf("ArrayNumNodes: ");
		for (int i = 0; i < NumTrees; i++)
		{
			printf("%d, ", ArrayNumNodes[i]);
		}
		printf("\n");
	}
	int writeToFile(char * string_file)
	{
		FILE * fp = fopen(string_file, "w");
		//
		fprintf(fp, "A RandomForestDecision Model with:\n");
		//
		fprintf(fp, "NumTrees: %d\n", NumTrees);
		fprintf(fp, "NumFeatures: %d\n", NumFeatures);
		fprintf(fp, "NumTypes: %d\n", NumTypes);
		//
		fprintf(fp, "ArrayNumNodes: ");
		for (int i = 0; i < NumTrees; i++)
		{
			fprintf(fp, "%d, ", ArrayNumNodes[i]);
		}
		fprintf(fp, "\n");
		//
		fprintf(fp, "ArrayTrees:\n");
		//
		for (int i = 0; i < NumTrees; i++)
		{
			ArrayTrees[i].writeToFile(fp);
		}
		//
		fprintf(fp, "RFD_End.\n");
		//
		fclose(fp);
		//
		return 0;
	}
	int loadFromFile(char * string_file)
	{
		FILE * fp = fopen(string_file, "r");
		//
		int LenBuff = 1024;
		char * buff = new char[LenBuff];
		char * str_begin;
		//
		fgets(buff, LenBuff, fp);   // 第一行
		//
		fgets(buff, LenBuff, fp);   // 第二行
		str_begin = strchr(buff, ':');
		sscanf(str_begin + 1, "%d", &NumTrees);
		//
		fgets(buff, LenBuff, fp);   // 第三行
		str_begin = strchr(buff, ':');
		sscanf(str_begin + 1, "%d", &NumFeatures);
		//
		fgets(buff, LenBuff, fp);   // 第四行
		str_begin = strchr(buff, ':');
		sscanf(str_begin + 1, "%d", &NumTypes);
		//
		fgets(buff, LenBuff, fp);   // 第五行，ArrayNumNodes
		fgets(buff, LenBuff, fp);   // 第五行，或第六行
		while (strchr(buff, ':') == NULL) fgets(buff, LenBuff, fp);   // 第五行，或第六行
		//
		delete [] buff;
		//
		this->setNumTrees(NumTrees);
		this->setNumInputOutput(NumFeatures, NumTypes);
		//
		// 读取每一棵树
		for (int i = 0; i < NumTrees; i++)
		{
			ArrayNumNodes[i] = ArrayTrees[i].loadFromFile(fp);
		}
		//
		fclose(fp);
		//
		return 0;
	}
	//

	// utilities
	int grow(FloatMat & Samples, FloatMat & Labels, float * ArrayFeatureStrides)
	{
		// paras for grow
		ArrayTrees[0].paras.copyFrom(paras);
		//
		int NumSamples;
		Samples.getMatSize(NumSamples, NumFeatures);
		Labels.getMatSize(NumSamples, NumTypes);
		//
		float * ArrayStrides = new float[NumFeatures];
		if (ArrayFeatureStrides == NULL)
		{
			for (int f = 0; f < NumFeatures; f++) ArrayStrides[f] = paras.MinStride;
		}
		else
		{
			for (int f = 0; f < NumFeatures; f++) ArrayStrides[f] = ArrayFeatureStrides[f];
		}
		//
		int num_samples, num_features;
		int * array_samples = new int[NumSamples];
		int * array_features = new int[NumFeatures];
		float * array_strides = new float[NumFeatures];
		//
		for (int i = 0; i < NumTrees; i++)
		{
			// 随机选择样本
			num_samples = 0;
			for (int r = 0; r < NumSamples; r++)
			{
				if ((rand()%100)/100 < SamplePortion)
				{
					array_samples[num_samples] = r;
					num_samples++;
				}
			}
			// 随机选择特征
			if (FeaturePortion > 1)
			{
				float ratio = 1.0 * FeaturePortion / NumFeatures;
				int NumFloor = (int) FeaturePortion;
				//
				num_features = 0;
				while (num_features != NumFloor)
				{
					num_features = 0;
					for (int r = 0; r < NumFeatures; r++)
					{
						if ((rand()%100)/100 < ratio)
						{
							array_features[num_features] = r;
							num_features++;
						}
					}
				}// while
			}
			else
			{
				num_features = 0;
				for (int r = 0; r < NumFeatures; r++)
				{
					if ((rand()%100)/100 < FeaturePortion)
					{
						array_features[num_features] = r;
						num_features++;
					}
				}
			}// if FeaturePortion
			//
			// 复制提取
			FloatMat SamplesPreUse(num_samples, NumFeatures);
			FloatMat samples_in_use(num_samples, num_features);
			float * labels_in_use = new float[num_samples];
			//
			float * labels_curr = NULL;
			int posi_s = 0;
			int LenS = sizeof(float) * NumFeatures;
			//
			for (int s = 0; s < num_samples; s++, posi_s += NumFeatures)
			{
				memcpy(SamplesPreUse.data + posi_s, Samples.data + array_samples[s]*NumFeatures, LenS);
				//
				labels_curr = Labels.data + array_samples[s]*NumTypes;
				//
				for (int t = 0; t < NumTypes; t++)
				{
					if (labels_curr[t] == 1)
					{
						labels_in_use[s] = t;
						break;
					}
				} // for t
			}
			//
			for (int f = 0; f < num_features; f++)
			{
				array_strides[f] = ArrayStrides[array_features[f]];
				//
				//cblas_scopy (const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy);
				cblas_scopy(num_samples, SamplesPreUse.data + array_features[f], NumFeatures, samples_in_use.data + f, num_features);
			}
			//

			// 单棵树生成
			ArrayTrees[i].grow(samples_in_use, labels_in_use, array_features, array_strides);
			//
			ArrayNumNodes[i] = ArrayTrees[i].NumNodes;
			//

		}// for i

		//
		delete [] ArrayStrides;
		delete [] array_samples;
		delete [] array_features;
		delete [] array_strides;
		//
		return 0;
	}
	//
	int predict(FloatMat & Samples, FloatMat & Results)
	{
		int NumSamples, NumTemp;
		Samples.getMatSize(NumSamples, NumTemp);
		//
		if (NumTemp != NumFeatures)
		{
			return -1;
		}
		//
		Results.setMatSize(NumSamples, NumTypes);
		Results.setMatConstant(0);
		float * data_results;
		//
		float * ArrayDecisions = new float[NumSamples];
		int type;
		//
		for (int i = 0; i < NumTrees; i++)
		{
			ArrayTrees[i].predict(Samples, ArrayDecisions);
			//
			data_results = Results.data;
			for (int s = 0; s < NumSamples; s++, data_results += NumTypes)
			{
				type = (int)(ArrayDecisions[s]);    //
				data_results[type]++;
			}// for s
		}//for i
		//
		Results = Results * (1.0/NumTrees);
		//

		return 0;
	}
	//

	//
	int scorePerformance(FloatMat & Results, FloatMat & Labels, int type, float Criteria, float * performance)
	{
		int TruePositives = 0;
		int PredictedPositives = 0;
		int TruePredictedPositives = 0;
		//
		int NumSamples, NumTypes;
		Labels.getMatSize(NumSamples, NumTypes);
		//
		float * data_label = Labels.data;
		float * data_result = Results.data;
		for (int s = 0; s < NumSamples; s++, data_label += NumTypes, data_result += NumTypes)
		{
			if (data_result[type] >= Criteria)
			{
				PredictedPositives++;
				if (data_label[type] == 1) TruePredictedPositives++;
			}
			//
			if (data_label[type] == 1) TruePositives++;

		}// for s

		//
		if (PredictedPositives > 0) performance[0] = TruePredictedPositives/PredictedPositives;
		else performance[0] = 0;
		//
		performance[1] = TruePredictedPositives/TruePositives;
		//
		performance[2] = TruePositives;
		performance[3] = PredictedPositives;
		performance[4] = TruePredictedPositives;
		//

		return 0;
	}
	//

};
//


#endif

