
//
#include <direct.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

//
#include "FloatMat.h"
#include "RFD_Model.h"
//

//
void loadConfiguration();
//
int FlagLoadFromFile;
int FlagTraining;
int FlagFiles;
//
int NumInput;
int NumOutput;
//
unsigned int SeedForRandom;
float PortionSamples;
float PortionFeatures;
//
int NumTrees;
int MaxDepth;
//
int TypeFocus;    //
float Criteria;    // 0.95, 0.85, 0
//
int MinInstances;
float MinInfoGain;
float MinStride;
int ImpurityType;
//

//
int main()
{  
	//printf("\n");
	printf("ApplicationRFD begin ...\n\n");
	//
	// direct
	mkdir("AutoRFD_working_direct");
	chdir("AutoRFD_working_direct");
	//
	char WORK_DIRECT[128];
	getcwd(WORK_DIRECT, sizeof(WORK_DIRECT)); 
	//
	// configuration
	loadConfiguration();
	//
	printf("Configuration loaded.\n\n");
	//
	printf("FlagLoadFromFile: %d\n", FlagLoadFromFile);
	printf("FlagTraining: %d\n", FlagTraining);
	printf("FlagFiles: %d\n", FlagFiles);
	printf("\n");
	//
	printf("NumInput: %d\n", NumInput);
	printf("NumOutput: %d\n", NumOutput);
	printf("\n");
	//
	printf("SeedForRandom: %d\n", SeedForRandom);
	printf("PortionSamples: %.4f\n", PortionSamples);
	printf("PortionFeatures: %.4f\n", PortionFeatures);
	printf("\n");
	//
	printf("NumTrees: %d\n", NumTrees);
	printf("MaxDepth: %d\n", MaxDepth);
	printf("TypeFocus: %d\n", TypeFocus);
	printf("Criteria: %.2f\n", Criteria);
	printf("\n");
	//
	printf("MinInstances: %d\n", MinInstances);
	printf("MinInfoGain: %.4f\n", MinInfoGain);
	printf("MinStride: %.4f\n", MinStride);
	printf("ImpurityType: %d\n", ImpurityType);
	printf("\n");
	//

	//
	RFD_Model rfd;
	rfd.setNumTrees(NumTrees);
	rfd.setNumInputOutput(NumInput, NumOutput);
	//
	ParasBDT paras;
	paras.MaxDepth = MaxDepth;
	paras.MinInstancesPerNode = MinInstances;
	paras.MinInfoGain = MinInfoGain;
	paras.MinStride = MinStride;
	paras.FlagImpurity = ImpurityType;
	//
	rfd.paras.copyFrom(paras);
	//
	rfd.SeedForRandom = SeedForRandom;
	rfd.PortionSamples = PortionSamples;
	rfd.PortionFeatures = PortionFeatures;
	//
	printf("initialized.\n");
	//
	getchar();
	//

	// files
	char TrainingSamples_Filename[32];
	char TrainingLabels_Filename[32];
	char TestSamples_Filename[32];
	char TestLabels_Filename[32];
	//
	char RFD_Filename[32];
	//
	if (FlagFiles == 0)
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels.txt");
		strcpy(TestSamples_Filename, "TestSamples.txt");
		strcpy(TestLabels_Filename, "TestLabels.txt");
		//
		strcpy(RFD_Filename, "RFD_File.txt");
	}
	else if (FlagFiles == 1)
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples_Ascend.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels_Ascend.txt");
		strcpy(TestSamples_Filename, "TestSamples_Ascend.txt");
		strcpy(TestLabels_Filename, "TestLabels_Ascend.txt");
		//
		strcpy(RFD_Filename, "RFD_File_Ascend.txt");
	}
	else if (FlagFiles == -1)
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples_Descend.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels_Descend.txt");
		strcpy(TestSamples_Filename, "TestSamples_Descend.txt");
		strcpy(TestLabels_Filename, "TestLabels_Descend.txt");
		//
		strcpy(RFD_Filename, "RFD_File_Descend.txt");
	}
	else
	{
		strcpy(TrainingSamples_Filename, "TrainingSamples.txt");
		strcpy(TrainingLabels_Filename, "TrainingLabels.txt");
		strcpy(TestSamples_Filename, "TestSamples.txt");
		strcpy(TestLabels_Filename, "TestLabels.txt");
		//
		strcpy(RFD_Filename, "RFD_File.txt");
	}

	// Load model
	if (FlagLoadFromFile == 1)
	{
		// load
		int iLoad = rfd.loadFromFile(RFD_Filename);   //
		if (iLoad == 0)
		{
			printf("Model Loaded from %s.\n", RFD_Filename);
		}
		else
		{
			printf("Error when loading model from %s.\n", RFD_Filename);
		}
		//
		getchar();
		//
	}
	else
	{
		printf("FlagLoadFromFile == 0.\n");
		printf("\n");
	}

	//
	float * performance = new float[5];
	//
	// Training
	if (FlagTraining == 1)
	{
		printf("FlagTraining == 1.\n\n");
		//
		// TrainingSamples
		FloatMat TrainingSamples(1, NumInput);
		TrainingSamples.loadAllDataInFile(TrainingSamples_Filename);
		printf("TrainingSamples loaded.\n");
		//
		int NumRows, NumCols;
		TrainingSamples.getMatSize(NumRows, NumCols);
		printf("TrainingSamples NumRows: %d\n", NumRows);
		//
		// TrainingLabels
		FloatMat TrainingLabels(1, NumOutput);
		TrainingLabels.loadAllDataInFile(TrainingLabels_Filename);
		printf("TrainingLabels loaded.\n");
		//
		TrainingLabels.getMatSize(NumRows, NumCols);
		printf("TrainingLabels NumRows: %d\n", NumRows);
		//
		getchar();
		//

		// Training Process
		//printf("\n");
		printf("Training Process:\n");
		//
		rfd.grow(TrainingSamples, TrainingLabels, NULL);
		//
		rfd.writeToFile(RFD_Filename);
		//
		FloatMat Results;
		rfd.predict(TrainingSamples, Results);
		rfd.scorePerformance(Results, TrainingLabels, TypeFocus, Criteria, performance);
		//
		printf("\n");
		printf("precision: %.4f\n", performance[0]);
		printf("recall: %.4f\n", performance[1]);
		printf("TruePositive: %.0f\n", performance[2]);
		printf("PredictedPositive: %.0f\n", performance[3]);
		printf("TruePredictedPositive: %.0f\n", performance[4]);
		printf("\n");
		//
		printf("Training Process Ended, Model saved.\n");
		//gfn.display();
		//
		//getchar();
		//
	}
	else
	{
		printf("FlagTraining == 0.\n\n");
		//
		// TestSamples
		FloatMat TestSamples(1, NumInput);
		TestSamples.loadAllDataInFile(TestSamples_Filename);
		printf("TestSamples loaded.\n");
		//
		int NumRows, NumCols;
		TestSamples.getMatSize(NumRows, NumCols);
		printf("TestSamples NumRows: %d\n", NumRows);
		//
		// TestLabels
		FloatMat TestLabels(1, NumOutput);
		TestLabels.loadAllDataInFile(TestLabels_Filename);
		printf("TestLabels loaded.\n");
		//
		TestLabels.getMatSize(NumRows, NumCols);
		printf("TestLabels NumRows: %d\n", NumRows);
		//
		getchar();
		//

		printf("Test Process ...\n");
		//
		FloatMat Results;
		rfd.predict(TestSamples, Results);
		rfd.scorePerformance(Results, TestLabels, TypeFocus, Criteria, performance);
		//
		printf("\n");
		printf("precision: %.4f\n", performance[0]);
		printf("recall: %.4f\n", performance[1]);
		printf("TruePositive: %.0f\n", performance[2]);
		printf("PredictedPositive: %.0f\n", performance[3]);
		printf("TruePredictedPositive: %.0f\n", performance[4]);
		printf("\n");
		//
		printf("Test Process Ended.\n");
		//
		//getchar();
		//
	}
	//

	//
	SeedForRandom = rfd.SeedForRandom;
	//
	char RFD_Backup_Filename[128];
	if (FlagFiles == 1)
	{
		sprintf(RFD_Backup_Filename, "RFD_File_Ascend_%.4f_%.4f_%d_%d_%.2f_%d.txt",
				performance[0], performance[1], NumTrees, MaxDepth, Criteria, SeedForRandom);
	}
	else if (FlagFiles == -1)
	{
		sprintf(RFD_Backup_Filename, "RFD_File_Descend_%.4f_%.4f_%d_%d_%.2f_%d.txt",
				performance[0], performance[1], NumTrees, MaxDepth, Criteria, SeedForRandom);
	}
	else
	{
		sprintf(RFD_Backup_Filename, "RFD_File_%.4f_%.4f_%d_%d_%.2f_%d.txt",
				performance[0], performance[1], NumTrees, MaxDepth, Criteria, SeedForRandom);
	}
	//
	rfd.writeToFile(RFD_Backup_Filename);
	//
	delete [] performance;
	//



	//
	printf("\n");
	printf("ApplicationRFD end.\n");

	getchar();
	return 0; 

}
//

//
void loadConfiguration()
{
	// Ä¬ÈÏÖµ
	FlagLoadFromFile = 0;
	FlagTraining = 0;
	FlagFiles = 0;
	//
	NumInput = 2;
	NumOutput = 2;
	//
	SeedForRandom = 0;
	PortionSamples = 0.2;
	PortionFeatures = 0.5;
	//
	NumTrees = 1;
	MaxDepth = 1;
	//
	TypeFocus = 0;
	Criteria = 0.50;
	//
	MinInstances = 1;
	MinInfoGain = 0.0001;
	MinStride = 0.1;
	ImpurityType = 0;
	//

	//
	FILE * fid = fopen("AutoGFN_Configuration.txt","r");

	if (fid == NULL)
	{
		fid = fopen("AutoGFN_Configuration.txt","w");

		fprintf(fid, "FlagLoadFromFile: %d\n", FlagLoadFromFile);
		fprintf(fid, "FlagTraining: %d\n", FlagTraining);
		fprintf(fid, "FlagFiles: %d\n", FlagFiles);
		//
		fprintf(fid, "NumInput: %d\n", NumInput);
		fprintf(fid, "NumOutput: %d\n", NumOutput);
		//
		fprintf(fid, "SeedForRandom: %d\n", SeedForRandom);
		fprintf(fid, "PortionSamples: %.4f\n", PortionSamples);
		fprintf(fid, "PortionFeatures: %.4f\n", PortionFeatures);
		//
		fprintf(fid, "NumTrees: %d\n", NumTrees);
		fprintf(fid, "MaxDepth: %d\n", MaxDepth);
		fprintf(fid, "TypeFocus: %d\n", TypeFocus);
		fprintf(fid, "Criteria: %.2f\n", Criteria);
		//
		fprintf(fid, "MinInstances: %d\n", MinInstances);
		fprintf(fid, "MinInfoGain: %.4f\n", MinInfoGain);
		fprintf(fid, "MinStride: %.4f\n", MinStride);
		fprintf(fid, "ImpurityType: %d\n", ImpurityType);
		//
		fprintf(fid, "End.");
		//
	}
	else
	{
		int LenBuff = 64;
		char * buff = new char[LenBuff];
		int curr;
		//
		while(fgets(buff, LenBuff, fid) != NULL)
		{
			if (strlen(buff) < 5) continue;
			//
			curr = 0;
			while (buff[curr] != ':') curr++;
			//
			buff[curr] = '\0';
			curr++;
			//
			if (strcmp(buff, "FlagLoadFromFile") == 0)         //
			{
				sscanf(buff + curr, "%d", &FlagLoadFromFile);
			}
			else if (strcmp(buff, "FlagTraining") == 0)
			{
				sscanf(buff + curr, "%d", &FlagTraining);
			}
			else if (strcmp(buff, "FlagFiles") == 0)
			{
				sscanf(buff + curr, "%d", &FlagFiles);
			}
			else if (strcmp(buff, "NumInput") == 0)   //
			{
				sscanf(buff + curr, "%d", &NumInput);
			}
			else if (strcmp(buff, "NumOutput") == 0)
			{
				sscanf(buff + curr, "%d", &NumOutput);
			}
			else if (strcmp(buff, "SeedForRandom") == 0)   //
			{
				sscanf(buff + curr, "%d", &SeedForRandom);
			}
			else if (strcmp(buff, "PortionSamples") == 0)   //
			{
				sscanf(buff + curr, "%f", &PortionSamples);
			}
			else if (strcmp(buff, "PortionFeatures") == 0)
			{
				sscanf(buff + curr, "%f", &PortionFeatures);
			}
			else if (strcmp(buff, "NumTrees") == 0)   //
			{
				sscanf(buff + curr, "%d", &NumTrees);
			}
			else if (strcmp(buff, "MaxDepth") == 0)
			{
				sscanf(buff + curr, "%d", &MaxDepth);
			}
			else if (strcmp(buff, "TypeFocus") == 0)   //
			{
				sscanf(buff + curr, "%d", &TypeFocus);
			}
			else if (strcmp(buff, "Criteria") == 0)
			{
				sscanf(buff + curr, "%f", &Criteria);
			}
			else if (strcmp(buff, "MinInstances") == 0)       //
			{
				sscanf(buff + curr, "%d", &MinInstances);
			}
			else if (strcmp(buff, "MinInfoGain") == 0)
			{
				sscanf(buff + curr, "%f", &MinInfoGain);
			}
			else if (strcmp(buff, "MinStride") == 0)
			{
				sscanf(buff + curr, "%f", &MinStride);
			}
			else if (strcmp(buff, "ImpurityType") == 0)
			{
				sscanf(buff + curr, "%d", &ImpurityType);
			}

		}// while fgets

		//
		delete [] buff;
	}

	fclose(fid);
	//
}
//

