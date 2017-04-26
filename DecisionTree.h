
#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "FloatMat.h"


//
class ParasBDT
{
public:
	// Paras for grow
	//
	int MaxDepth;
	int MinInstancesPerNode;
	float MinInfoGain;
	//
	float MinStride;
	int FlagImpurity;
	//
	static const int DT_GINI = 0;
	static const int DT_ENT = 1;  // entropy
	static const int DT_VAR = 2;  // variance
	//
	ParasBDT()
	{
		MaxDepth = 10;
		MinInstancesPerNode = 1;
		MinInfoGain = 0.0001;
		//
		MinStride = 0.1;
		FlagImpurity = DT_GINI;
	}
	//
	void resetParasBDT()
	{
		MaxDepth = 10;
		MinInstancesPerNode = 1;
		MinInfoGain = 0.0001;
		//
		MinStride = 0.1;
		FlagImpurity = DT_GINI;

	}
	//
	void copyFrom(ParasBDT paras)
	{
		MaxDepth = paras.MaxDepth;
		MinInstancesPerNode = paras.MinInstancesPerNode;
		MinInfoGain = paras.MinInfoGain;
		//
		MinStride = paras.MinStride;
		FlagImpurity = paras.FlagImpurity;
	}
	//
};
//
class NodeBDT
{
public:
	int id;
	int depth;
	//
	int feature_index;
	float value_split;
	//
	NodeBDT * left;
	NodeBDT * right;
	//
	// 当为叶结点时，value_split 为输出，
	// 通过 left == NULL 来判断是否为叶节点，
	// 也可通过 feature_index == -1 来判断，
	//
	// 最初结点id = 0, depth = 0;
	//

	//
	NodeBDT()
	{
		id = 0;
		depth = 0;
		feature_index = -1;
		value_split = 0;
		left = NULL;
		right = NULL;
	}
	~NodeBDT()
	{
		if (right != NULL) delete right;
		if (left != NULL) delete  left;
	}
	//
	float decide(float * array)
	{
		// 叶节点
		if (left == NULL) return value_split;
		//
		// 枝节点
		if (array[feature_index] <= value_split) return left->decide(array);
		else return right->decide(array);
	}
	//
	int grow(int & CountNodes, ParasBDT & paras,
			FloatMat & Samples, float * ArrayTargets, int * ArrayFeatures, float * ArrayStrides)
	{
		id = CountNodes;
		CountNodes++;

		// for leaf check
		int FlagLeaf = 0;
		//
		// 最大深度
		if (depth == paras.MaxDepth) FlagLeaf = -1;
		//
		int num_samples, num_features;
		Samples.getMatSize(num_samples, num_features);
		//
		// 最小样本数
		if (num_samples <= paras.MinInstancesPerNode) FlagLeaf = -1;
		//
		// 信息增益，找出最好分割
		int num_left = 0;
		int num_right = 0;
		int feature_best = -1;
		//
		if (FlagLeaf == 0)
		{
			float info_gain = searchBestSplit(Samples, ArrayTargets, ArrayStrides,
					feature_best, value_split, num_left, num_right);
			//
			if (info_gain <= paras.MinInfoGain) FlagLeaf = -1;

			//printf("info_gain: %f\n", info_gain);
			//printf("num_left, num_right: %d, %d\n", num_left, num_right);
		}

		//
		// 叶节点
		if (FlagLeaf == -1)
		{
			feature_index = -1;
			value_split = computeLeafOutput(ArrayTargets, num_samples);
			//
			left = NULL;
			right = NULL;
			//
			return CountNodes;
		}
		//
		// 未到叶节点
		feature_index = ArrayFeatures[feature_best];
		//
		FloatMat samples_left(num_left, num_features);
		FloatMat samples_right(num_right, num_features);
		float * targets_left = new float[num_left];
		float * targets_right = new float[num_right];
		//
		splitSamplesAndTargets(Samples, ArrayTargets, feature_best, value_split,
				samples_left, samples_right, targets_left, targets_right);
		//
		// 增加左节点
		left = new NodeBDT;
		left->depth = depth + 1;
		left->grow(CountNodes, paras, samples_left, targets_left, ArrayFeatures, ArrayStrides);

		// 增加右节点
		right = new NodeBDT;
		right->depth = depth + 1;
		right->grow(CountNodes, paras, samples_right, targets_right, ArrayFeatures, ArrayStrides);

		//
		return CountNodes;
	}
	//

	//
	float searchBestSplit(FloatMat & Samples, float * ArrayTypes, float * ArrayStrides,
			int & feature_best, float & value_best, int & num_left, int & num_right)
	{
		int num_samples, num_features;
		Samples.getMatSize(num_samples, num_features);
		//
		float impurity_before = computeImpurity(ArrayTypes, num_samples);
		//

		// sort and search
		float info_gain_max = 0;
		float impurity_temp, info_gain_temp;
		//
		float * values = new float[num_samples];
		float * types = new float[num_samples];
		float value_curr;
		//
		for (int f = 0; f < num_features; f++)
		{
			//cblas_scopy (const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy);
			cblas_scopy(num_samples, Samples.data + f, num_features, values, 1);
			//void * memcpy(void * dest, const void * src, size_t n);
			memcpy(types, ArrayTypes, sizeof(float) * num_samples);
			// sort
			quickSortAssociated(values, types, 0, num_samples - 1);

			//
			value_curr = values[0] + ArrayStrides[f]/2;
			//
			for (int s = 0; s < num_samples; s++)
			{
				if (values[s] <= value_curr) continue;    //
				//
				num_right = num_samples - s;
				//
				impurity_temp = computeImpurity(types, s) * s;
				impurity_temp += computeImpurity(types + s, num_right) * num_right;
				impurity_temp /= num_samples;
				//
				info_gain_temp = impurity_before - impurity_temp;
				//info_gain_temp *=  1.0 * (num_samples/s) * (num_samples/num_right);
				//
				if (info_gain_temp > info_gain_max)
				{
					info_gain_max = info_gain_temp;
					feature_best = f;
					value_best = value_curr;
					num_left = s;
				}

				//
				value_curr += ArrayStrides[f];
				while (value_curr <= values[s]) value_curr += ArrayStrides[f];
				//
			} // for s

		}// for f
		//
		delete [] values;
		delete [] types;
		//

		//
		num_right = num_samples - num_left;
		//
		return info_gain_max;
	}
	//
	float splitSamplesAndTargets(FloatMat & Samples, float * ArrayTargets, int & feature_best, float & value_best,
			FloatMat & samples_left, FloatMat & samples_right, float * array_left, float * array_right)
	{
		int num_samples, num_features;
		Samples.getMatSize(num_samples, num_features);

		// split
		float * data_left = samples_left.data;
		float * data_right = samples_right.data;
		int posi_left = 0;
		int posi_right = 0;
		//
		float * data_sample = Samples.data;
		float* data_value = Samples.data + feature_best;
		//
		int LenCopy = sizeof(float) * num_features;
		//
		for (int s = 0; s < num_samples; s++, data_sample += num_features, data_value += num_features)
		{
			if (*data_value <= value_best)
			{
				memcpy(data_left, data_sample, LenCopy);
				data_left += num_features;
				//
				array_left[posi_left] = ArrayTargets[s];
				posi_left++;
			}
			else
			{
				memcpy(data_right, data_sample, LenCopy);
				data_right += num_features;
				//
				array_right[posi_right] = ArrayTargets[s];
				posi_right++;
			}
		}// for s

		return 0;
	}
	//

	//
	float computeLeafOutput(float * array_types, int num_samples)
	{
		float * array_unique = new float[num_samples];
		int * array_count = new int[num_samples];
		//
		int num_unique = 1;
		array_unique[0] = array_types[0];
		array_count[0] = 1;
		//
		int flag_same;
		for (int s = 1; s < num_samples; s++)
		{
			flag_same = 0;
			//
			for (int i = 0; i < num_unique; i++)
			{
				if (array_types[s] == array_unique[i])
				{
					flag_same = 1;
					array_count[i]++;
					break;
				}
			}// for i
			//
			if (flag_same == 0)
			{
				array_unique[num_unique] = array_types[s];
				num_unique++;
			}

		}// for s

		//
		float output = array_unique[0];
		int num_max = array_count[0];
		//
		for (int i = 1; i < num_unique; i++)
		{
			if (array_count[i] > num_max)  //
			{
				output = array_unique[i];
				num_max = array_count[i];
			}
			else if (array_count[i] == num_max)   //
			{
				if (array_unique[i] > output) output = array_unique[i];
			}
		}// for i

		//
		delete [] array_unique;
		delete [] array_count;
		//
		return output;
	}
	//
	float computeImpurity(float * array_types, int num_samples)
	{
		if (num_samples <= 1) return 0;

		//
		float * array_unique = new float[num_samples];
		int * array_count = new int[num_samples];
		//
		int num_unique = 1;
		array_unique[0] = array_types[0];
		array_count[0] = 1;
		//
		int flag_same;
		for (int s = 1; s < num_samples; s++)
		{
			flag_same = 0;
			//
			for (int i = 0; i < num_unique; i++)
			{
				if (array_types[s] == array_unique[i])
				{
					flag_same = 1;
					array_count[i]++;
					break;
				}
			}// for i
			//
			if (flag_same == 0)
			{
				array_unique[num_unique] = array_types[s];
				array_count[num_unique] = 1;
				//
				num_unique++;
			}

		}// for s

		//
		float imp = 0;
		float prob;
		//
		for (int i = 0; i < num_unique; i++)
		{
			prob = 1.0 * array_count[i] / num_samples;
			imp += (1- prob) * prob;
		}

		//
		delete [] array_unique;
		delete [] array_count;
		//
		return imp;
	}
	//
	void quickSortAssociated(float * array, float * related, int left, int right)
	{
		if (left >= right) return;

		int i = left;
		int j = right;
		float key = array[left];
		float key_rel = related[left];

		//
		while (i < j)
		{
			while (i < j && key <= array[j]) j--;

			array[i] = array[j];
			related[i] = related[j];

			while (i < j && key >= array[i]) i++;

			array[j] = array[i];
			related[j] = related[i];

		}
		//
		array[i] = key;
		related[i] = key_rel;
		//
		quickSortAssociated(array, related, left, i - 1);
		quickSortAssociated(array, related, i + 1, right);
		//
	}
	//

	//
	int writeNodesToFile(FILE * fp, int & CountNodes)
	{
		fprintf(fp, "%d, %d, %d, %f, ", id, depth, feature_index, value_split);
		CountNodes++;
		//
		if (left == NULL) return CountNodes;
		//
		left->writeNodesToFile(fp, CountNodes);
		right->writeNodesToFile(fp, CountNodes);
		//
		return CountNodes;
	}
	int loadNodesFromFile(FILE * fp, int & CountNodes)
	{
		// 读取一个节点的数据
		int LenBuff = 128;
		char * buff = new char[LenBuff];
		char ch;
		//
		int Posi = 0;
		int NumRead = 0;
		while((ch = fgetc(fp)) != EOF)
		{
			buff[Posi] = ch;
			Posi++;
			//
			if (ch == ',') NumRead++;
			if (NumRead == 4) break;
		}
		// 解析数据
		char * str_begin = buff;
		char * str_end = strchr(str_begin, ',');
		*str_end = '\0';
		sscanf(str_begin, "%d", &id);
		//
		str_begin = str_end + 1;
		str_end = strchr(str_begin, ',');
		*str_end = '\0';
		sscanf(str_begin, "%d", &depth);
		//
		str_begin = str_end + 1;
		str_end = strchr(str_begin, ',');
		*str_end = '\0';
		sscanf(str_begin, "%d", &feature_index);
		//
		str_begin = str_end + 1;
		str_end = strchr(str_begin, ',');
		*str_end = '\0';
		sscanf(str_begin, "%f", &value_split);
		//
		delete [] buff;
		//
		CountNodes++;
		//

		//
		// 到达叶节点
		if (feature_index == -1) return CountNodes;
		//
		// 未到叶节点
		//
		// 增加左节点
		left = new NodeBDT;
		left->loadNodesFromFile(fp, CountNodes);

		// 增加右节点
		right  = new NodeBDT;
		right->loadNodesFromFile(fp, CountNodes);

		//
		return CountNodes;
	}
	//

};
//

//
class DecisionTree
{
public:
	//
	NodeBDT * NodeRoot;
	int NumNodes;
	//
	// 遍历采用：根->左->右序，
	//
	ParasBDT paras;
	//

	//
	DecisionTree(void)
	{
		NumNodes = 1;
		NodeRoot = new NodeBDT;
	}
	~DecisionTree()
	{
		delete NodeRoot;
	}
	//

	//
	int predict(FloatMat & Samples, float * ArrayResults)
	{
		int NumSamples, NumFeatures;
		Samples.getMatSize(NumSamples, NumFeatures);
		//
		float * array_instance = Samples.data;
		for (int s = 0; s < NumSamples; s++, array_instance += NumFeatures)
		{
			ArrayResults[s] = NodeRoot->decide(array_instance);
		}// for s

		return NumSamples;
	}
	//
	int grow(FloatMat & Samples, float * ArrayTargets, int * ArrayFeatures, float * ArrayStrides)
	{
		NumNodes = 0;
		NodeRoot->grow(NumNodes, paras, Samples, ArrayTargets, ArrayFeatures, ArrayStrides);
		//
		return NumNodes;
	}
	//

	//
	int writeToFile(FILE * fp)
	{
		fprintf(fp, "A BinaryDecisionTree with NumNodes: %d\n", NumNodes);
		//
		int CountNodes = 0;
		NodeRoot->writeNodesToFile(fp, CountNodes);
		fprintf(fp, "\n");
		//
		fprintf(fp, "BinaryDecisionTreeEnd: %d\n", CountNodes);
		//
		return CountNodes;
	}
	int loadFromFile(FILE * fp)
	{
		int LenBuff = 64;
		char * buff = new char[LenBuff];
		fgets(buff, LenBuff, fp);   // 第一行
		//
		// 节点信息
		int CountNodes = 0;
		NodeRoot->loadNodesFromFile(fp, CountNodes);
		//
		fgets(buff, LenBuff, fp);   // 最后一行，
		while (strchr(buff, ':') == NULL) fgets(buff, LenBuff, fp);   // 最后一行，
		//
		delete [] buff;
		//
		return CountNodes;

	}
	void display()
	{
		printf("A BinaryDecisionTree with NumNodes: %d\n", NumNodes);
	}
	//

};
//


#endif

