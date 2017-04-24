
#ifndef FLOAT_MAT_H
#define FLOAT_MAT_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <cblas.h>
#include <lapacke.h>

class FloatMat
{
private:
	int NumRows;
	int NumCols;
	//
	int NumTotal;

public:
	float * data;


	// 构造函数(无参，一般，复制) //(类型转换)
	FloatMat(void)
	{
		NumRows = 1;
		NumCols = 1;
		NumTotal = 1;

		data = new float[1];
		data[0] = 0;
	}
	FloatMat(int a, int b)
	{
		//if (a == 0 || b == 0) {}

		//
		NumRows = a;
		NumCols = b;
		NumTotal = a*b;

		data = new float[NumTotal];
		for (int i = 0; i < NumTotal; i++)
		{
			data[i] = 0;
		}
	}
	FloatMat(const FloatMat & mat)
	{
		NumRows = mat.NumRows;
		NumCols = mat.NumCols;
		NumTotal = mat.NumTotal; 

		data = new float[NumTotal];
		memcpy(data, mat.data, sizeof(float) * NumTotal);
	}
	// 析构函数
	~FloatMat()
	{
		delete [] data;
	}
	//
	// 结构信息函数
	void setMatSize(int a, int b)
	{
		NumRows = a;
		NumCols = b;
		NumTotal = a*b;

		delete [] data;
		data = new float[NumTotal];
		for (int i = 0; i < NumTotal; i++)
		{
			data[i] = 0;
		}
	}
	void getMatSize(int & a, int & b)
	{
		a = NumRows;
		b = NumCols;
	}
	int getNumTotal()
	{
		return NumTotal;
	}
	//
	// 赋值运算
	// operator=
	FloatMat & operator= (const FloatMat & mat)
	{
		//
		if (this == &mat) return *this; // 解决自我赋值的问题

		//
		NumRows = mat.NumRows;
		NumCols = mat.NumCols;
		NumTotal = mat.NumTotal;
		//
		delete [] data;
		data = new float[NumTotal];
		memcpy(data, mat.data, sizeof(float) * NumTotal);

		return *this;
	}
	//

	// 算术运算，BLAS
	// operator+
	FloatMat operator+(FloatMat mat)
	{
		FloatMat answ(NumRows, NumCols);
		//
		memcpy(answ.data, data, sizeof(float) * NumTotal);
		//
		cblas_saxpy (NumTotal, 1.0, mat.data, 1, answ.data, 1);
		//
		return answ;
	}	
	// operator-
	FloatMat operator-(FloatMat mat)
	{
		FloatMat answ(NumRows, NumCols);
		//
		memcpy(answ.data, data, sizeof(float) * NumTotal);
		//
		cblas_saxpy (NumTotal, -1.0, mat.data, 1, answ.data, 1);
		//
		return answ;
	}
	// operator*
	FloatMat operator*(float a)
	{
		FloatMat answ(NumRows, NumCols);
		//
		memcpy(answ.data, data, sizeof(float) * NumTotal);
		//
		cblas_sscal (NumTotal, a, answ.data, 1);
		//
		return answ;
	}	
	// operator*
	FloatMat operator*(FloatMat mat)
	{
		int answNumCols = mat.NumCols;
		//
		FloatMat answ(NumRows, answNumCols);
		//
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumRows, answNumCols, NumCols,
				1.0, data, NumCols, mat.data, answNumCols, 0.0, answ.data, answNumCols);
		//

		return answ;
	}
	// transpose
	FloatMat transpose()
	{
		FloatMat answ(NumCols, NumRows);

		//
		float * data_a = data;
		float * data_b = answ.data;
		//
		for (int i = 0; i < NumRows; i++, data_a += NumCols, data_b++)
		{
			//cblas_scopy (const MKL_INT n, const float *x, const MKL_INT incx, float *y, const MKL_INT incy);
			//
			// copy and transpose
			//
			cblas_scopy(NumCols, data_a, 1, data_b, NumRows);
			//
		}

		return answ;
	}
	//

	// 算术运算，No BLAS
	// plus alpha * I
	FloatMat plusWeightedIdentity(float alpha)
	{
		FloatMat answ(NumRows, NumCols);
		//
		int Posi = 0;
		int Step = NumCols + 1;
		//
		for (int i = 0; i < NumRows; i++)
		{
			answ.data[Posi] += alpha;
			//
			Posi += Step;
		}
		//
		return answ;

	}
	// multiplication element-wise
	FloatMat mul(FloatMat mat)
	{
		FloatMat answ(NumRows, NumCols);

		float * data2 = mat.data;

		for (int i = 0; i < NumTotal; i++)
		{
			answ.data[i] = data[i] * data2[i];
		}

		return answ;
	}
	// inverse
	FloatMat inverse(float alpha)
	{
		// 复制原矩阵，修改对角元，
		FloatMat mat(*this);
		//
		if (alpha != 0)
		{
			int posi = 0;
			int Step = NumCols + 1;
			for (int i = 0; i < NumRows; i++)
			{
				mat.data[posi] += alpha;

				//
				posi += Step;
			}
		}
		//

		// 矩阵，增广部分，
		FloatMat answ(NumRows, NumCols);
		answ.setMatConstant(0);
		//
		int posi = 0;
		int Step = NumCols + 1;
		for (int i = 0; i < NumRows; i++)
		{
			answ.data[posi] = 1;

			//
			posi += Step;
		}
		//

		//mat.display();
		//answ.display();
		//getchar();


		//
		float * data_mat = mat.data;
		float * data_answ = answ.data;
		//

		//顺序消元，生成上三角矩阵，
		float factor, value_diag;
		int posi_target;
		int posi_base;
		//
		for (int i = 0; i < NumRows; i++)
		{
			posi_base = i*NumCols + i;
			//
			value_diag = data_mat[posi_base];

			for (int a = i+1; a < NumRows; a++)
			{
				posi_target = a*NumCols + i;
				//
				factor = data_mat[posi_target]/value_diag;

				//原矩阵
				data_mat[posi_target] = 0;
				posi_base++;
				posi_target++;
				//
				for (int j = i+1; j < NumCols; j++)
				{
					data_mat[posi_target] -= factor * data_mat[posi_base];
					//
					posi_base++;
					posi_target++;

				}// for j

				//增广部分
				posi_base = i * NumCols;
				posi_target = a * NumCols;
				//
				for (int j = 0; j < NumCols; j++)
				{
					data_answ[posi_target] -= factor * data_answ[posi_base];
					//
					posi_base++;
					posi_target++;

				}// for j

			}// for a
		}// for i


		//逆序消元，生成单位矩阵，
		for (int i = NumRows - 1; i >= 0; i--)
		{
			posi_base = i*NumCols + i;
			//
			value_diag = data_mat[posi_base]; 

			//对角元
			//
			//原矩阵
			data_mat[posi_base] = 1;
			//
			//增广部分
			posi_target = i * NumCols;
			for (int j = 0; j < NumCols; j++)
			{
				data_answ[posi_target] /= value_diag;

				posi_target++;
			}

			//其它行			
			for (int a = i-1; a >= 0; a--)
			{
				posi_target = a*NumCols + i;
				//
				factor = data_mat[posi_target];

				//原矩阵
				data_mat[posi_target] = 0;
				//
				//增广部分
				posi_base = i * NumCols;
				posi_target = a * NumCols;
				for (int j = 0; j < NumCols; j++)
				{
					data_answ[posi_target] -= factor * data_answ[posi_base];
					//
					posi_base++;
					posi_target++;

				}// for j

			}// for a
		}// for i

		//
		return answ;

	}
	//

	// signs
	FloatMat getSigns()
	{
		FloatMat answ(NumRows, NumCols);

		for (int i = 0; i < NumTotal; i++)
		{
			if (data[i] > 0) answ.data[i] = 1;                      // 1
			else if (data[i] < 0) answ.data[i] = -1;            // -1
			else answ.data[i] = 0;                                      // 0
		}

		return answ;
	}
	// 常数
	void setMatConstant(float a)
	{
		for (int i = 0; i < NumTotal; i++)
		{
			data[i] = a;
		}
	}
	// 复制
	void copyFrom(FloatMat mat)
	{
		delete [] data;

		//
		NumRows = mat.NumRows;
		NumCols = mat.NumCols;
		NumTotal = mat.NumTotal; 

		data = new float[NumTotal];
		memcpy(data, mat.data, sizeof(float) * NumTotal);
	}
	// 随机化
	void randomize(int a, int b)
	{
		int d = (b - a)*100;
		for (int i = 0; i < NumTotal; i++)
		{
			data[i] = a + (rand()%d)/100.0;
		}
	}
	//
	void abs()
	{
		for (int i = 0; i < NumTotal; i++)
		{
			if (data[i] < 0) data[i] = -data[i];
		}
	}
	//
	void normalizeRows()
	{
		float sum;
		int posi, PosiCut = 0;

		for (int i = 0; i < NumRows; i++)
		{
			sum = 0;

			//
			posi = PosiCut;
			PosiCut += NumCols;

			for (; posi < PosiCut; posi++)
			{
				sum += data[posi];				
			}

			//
			posi = NumCols * i;

			for (; posi < PosiCut; posi++)
			{
				data[posi] /= sum;				
			}
		}
	}
	//
	FloatMat sumRows()
	{
		FloatMat answ(NumRows, 1);

		float sum;
		int posi, PosiCut = 0;

		for (int i = 0; i < NumRows; i++)
		{
			sum = 0;

			//
			posi = PosiCut;
			PosiCut += NumCols;

			for (; posi < PosiCut; posi++)
			{
				sum += data[posi];				
			}

			//
			answ.data[i] = sum;			
		}

		return answ;
	}
	FloatMat sumCols()
	{
		FloatMat answ(1, NumCols);

		int Posi;
		float sum;

		for (int j = 0; j < NumCols; j++)
		{
			sum = 0;
			Posi = j;
			for (int i = 0; i < NumRows; i++)
			{
				sum += data[Posi];
				Posi += NumCols;
			}

			//
			answ.data[j] = sum;

		}//for j

		return answ;
	}
	//
	float sumElementsAll()
	{
		float answ = 0;

		for (int i = 0; i < NumTotal; i++)
		{
			answ += data[i];
		}

		return answ;
	}
	float meanElementsAll()
	{
		float answ = 0;

		for (int i = 0; i < NumTotal; i++)
		{
			answ += data[i];
		}

		return answ/NumTotal;
	}
	//

	// 显示，文件读写
	void display()
	{
		int posi = 0;

		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				printf("%.6f, ", data[posi]);

				posi++;
			}

			printf("\n");
		}
	}
	void writeToFile(FILE * fid)
	{
		int posi = 0;

		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				fprintf(fid, "%f,", data[posi]);

				posi++;
			}

			fprintf(fid,"\n");
		}
	}
	void loadFromFile(FILE * fid, int NumRows)
	{
		// 2048, 4096
		int LenBuff = 4096;

		char * buff = new char[LenBuff];
		char * str_begin;
		int curr;

		//
		int Posi = 0;

		for (int i = 0; i < NumRows; i++)
		{
			fgets(buff, LenBuff, fid);

			//
			str_begin = buff;

			curr = 0;
			while (buff[curr] != '\n')
			{
				if (buff[curr] == ',')
				{
					buff[curr] = '\0';

					sscanf(str_begin, "%f", data + Posi);

					//
					Posi++;

					//
					curr++;

					str_begin = buff + curr;
				}
				else
				{
					curr++;
				}
			}

		}// for i NumRows

		//
		delete [] buff;
	}
	//
	int loadAllDataInFile(char * filepath)
	{
		//
		FILE * fid = fopen(filepath, "r");
		if (fid == NULL) return -1;

		// 2048, 4096
		int LenBuff = 4096;
		char * buff = new char[LenBuff];
		//

		//
		int Count = 0;
		//
		while (fgets(buff, LenBuff, fid) != NULL)
		{
			Count++;
		}
		fclose(fid);
		//

		//
		this->setMatSize(Count, NumCols);
		//
		fid = fopen(filepath, "r");
		this->loadFromFile(fid, Count);
		fclose(fid);
		//

		//
		return Count;

	}
	//

	// CBLAS
	float getNormL1()
	{
		return cblas_sasum (NumTotal, data, 1);   // abs, then sum
	}
	float getNormL2()
	{
		return cblas_snrm2 (NumTotal, data, 1);  // square, then sum, then square-root,
	}
	//
	float dotAndSumElementsAll(FloatMat mat)
	{
		return cblas_sdot (NumTotal, data, 1, mat.data,1);
	}
	//
	int getPosiMaxAmplitude()
	{
		return cblas_isamax (NumTotal, data, 1);
	}
	int getPosiMinAmplitude()
	{
		//return cblas_isamin (NumTotal, data, 1);
		//
		printf("WARNING: cblas_isamin NOT implemented in OpenBLAS\n");
		//
		return 0;
		//
	}
	//

	// SolveLinearEquations BLAS
	//
	int solveWithSymMat(FloatMat symMat, FloatMat & X)
	{
		//lapack_int LAPACKE_ssysv (int matrix_layout , char uplo , lapack_int n , lapack_int
		//nrhs , float * a , lapack_int lda , lapack_int * ipiv , float * b , lapack_int ldb );
		//
		// solve A * X = B;
		//
		// int iResult = B.solveWithSymMat(A, X);
		//
		// 复制
		FloatMat A(symMat);
		//
		X.copyFrom(*this);
		//
		// 求解
		int * ipiv = new int[NumRows];
		//
		int iRet = LAPACKE_ssysv(LAPACK_ROW_MAJOR, 'U', NumRows, NumCols, A.data, A.NumCols, ipiv, X.data, NumCols);
		//
		delete [] ipiv;
		//
		return iRet;

	}
	//
	int solveWithSymMatX(FloatMat symMat, FloatMat & X)
	{
		//lapack_int LAPACKE_ssysvx( int matrix_layout, char fact, char uplo, lapack_int n,
		//	lapack_int nrhs, const float* a, lapack_int lda, float* af, lapack_int ldaf,
		//	lapack_int* ipiv, const float* b, lapack_int ldb, float* x, lapack_int ldx, float*
		//	rcond, float* ferr, float* berr );

		//
		// solve A * X = B;
		//
		// int iResult = B.solveWithSymMatX(A, X);
		//
		// 复制
		FloatMat A(symMat);
		//
		X.setMatSize(NumRows, NumCols);
		//
		// 求解
		int * ipiv = new int[NumRows];
		float * af = new float[NumRows * NumRows];
		float * rcond = new float[1];
		float * ferr = new float[NumCols];
		float * berr = new float[NumCols];
		//
		int iRet = LAPACKE_ssysvx(LAPACK_ROW_MAJOR, 'N', 'U', NumRows, NumCols,
				A.data, A.NumCols, af, NumRows, ipiv, data, NumCols, X.data, NumCols, rcond, ferr, berr);
		//
		delete [] ipiv;
		delete [] af;
		delete [] rcond;
		delete [] ferr;
		delete [] berr;
		//
		return iRet;

	}
	//
	int solveWithSymMatXX(FloatMat symMat, FloatMat & X)
	{
		//lapack_int LAPACKE_ssysvxx( int matrix_layout, char fact, char uplo, lapack_int n,
		//lapack_int nrhs, float* a, lapack_int lda, float* af, lapack_int ldaf, lapack_int*
		//ipiv, char* equed, float* s, float* b, lapack_int ldb, float* x, lapack_int ldx,
		//float* rcond, float* rpvgrw, float* berr, lapack_int n_err_bnds, float* err_bnds_norm,
		//float* err_bnds_comp, lapack_int nparams, const float* params );

		//
		// solve A * X = B;
		//
		// int iResult = B.solveWithSymMatXX(A, X);
		//
		// 复制
		FloatMat A(symMat);
		//
		X.setMatSize(NumRows, NumCols);
		//
		// 求解
		int * ipiv = new int[NumRows];
		float * af = new float[NumRows * NumRows];
		float * rcond = new float[1];
		float * ferr = new float[NumCols];
		float * berr = new float[NumCols];
		//
		int iRet = LAPACKE_ssysvx(LAPACK_ROW_MAJOR, 'N', 'U', NumRows, NumCols,
				A.data, A.NumCols, af, NumRows, ipiv, data, NumCols, X.data, NumCols, rcond, ferr, berr);
		//
		delete [] ipiv;
		delete [] af;
		delete [] rcond;
		delete [] ferr;
		delete [] berr;
		//
		return iRet;

	}
	//

	//
	int solveWithSymMat_Self(FloatMat symMat, FloatMat & X)
	{
		//
		X = symMat.inverse(0) * (*this);

		//
		return 0;

	}
	//


};

//



#endif

