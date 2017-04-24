
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

	//
	FloatMat Mat;
	Mat.setMatSize(2, 3);
	Mat.setMatConstant(1);
	Mat.display();
	//
	FloatMat Mat1;
	Mat1.setMatSize(3, 2);
	Mat1.setMatConstant(1);
	Mat1.display();
	//
	FloatMat Mat2;
	Mat2 = Mat * Mat1;
	Mat2.display();

	//
	DecisionTree dt;// = new DecisionTree;
	dt.display();

	RFD_Model rfd;
	rfd.setNumTrees(2);
	rfd.setNumInputOutput(10, 2);
	rfd.display();
	//
	rfd.writeToFile("test.txt");


	//
	printf("\n");
	printf("ApplicationRFD end.\n");

	getchar();
	return 0; 

}
//
