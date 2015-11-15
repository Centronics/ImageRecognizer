#include <limits>

//#define MATRIXCONTROL
//#define BLOCKALG
#define FLOATSIGN
typedef double SignType;
typedef unsigned int LengthType;
#define SIGNMAX 5000.0
#define SIGNMIN	1.0
#define LENGTHMAX UINT_MAX

#include "MatrixStarter.cuh"
#include "VectorOfFiles.h"

const LengthType Iterations = 4;
const LengthType CtxLen     = 3; //Тормозит сильнее всех
const SignType   Addition   = 0;
const wchar_t*   BMPPics    = L"D:\\tst";
const wchar_t*   BMPPicsComp= L"D:\\tstComp";

static void** GetPicsMas(const wchar_t* BMPPics, vector<SMatrix*>*& _MatrVec);
static void DeleteMatrix(vector<SMatrix*>* MatrVec);

int main()
{
	void** Pars = nullptr;
	try
	{
		vector<SMatrix*>* MatrVecComp = nullptr;
		Pars = GetPicsMas(BMPPicsComp, MatrVecComp);
		TestWithCuda(Pars, MatrVecComp->size(), Iterations, CtxLen, true, Addition);
		delete[] Pars; Pars = nullptr;
		vector<SMatrix*>* MatrVec = nullptr;
		Pars = GetPicsMas(BMPPics, MatrVec);
		LengthType* const tstResults = TestWithCuda(Pars, MatrVec->size(), Iterations, CtxLen, false, Addition);
		delete[] Pars; Pars = nullptr;
		if(tstResults == 0)
			throw std::exception("Cant get results");
		cout << endl << "Results: " << endl << endl;
		for(size_t k = 0; k < MatrVec->size(); k++)
			cout << (*MatrVec)[k]->GetMatrixName() << " => " << MatrVecComp->at(tstResults[k])->GetMatrixName() << endl;
		free(tstResults);
		DeleteMatrix(MatrVec);
		DeleteMatrix(MatrVecComp);
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
		getchar();
		return 0;
	}
	catch(std::exception& ex)
	{
		cout << endl << ex.what() << endl;
	}
	catch(...)
	{
		cout << endl << "Unknown error" << endl;
	}
	getchar();
	delete[] Pars;
	return 1;
}

static void** GetPicsMas(const wchar_t* BMPPics, vector<SMatrix*>*& _MatrVec)
{
	static unsigned int LoadCount = 0;
	cout << endl << "Loading " << "step " << ++LoadCount << ":" << endl << endl;
	vector<SMatrix*>* MatrVec = GetMatrixes(BMPPics);
	if(MatrVec == nullptr) {
		throw std::exception("Cant get matrixes");
		return nullptr;
	}
	_MatrVec = MatrVec;
	vector<SMatrix*>::iterator MatrIter = MatrVec->begin();
	size_t Matrs = static_cast<size_t>(MatrVec->size());
	void** const Pars = new void*[Matrs * 2];
	void** ParIter = Pars;
	while(MatrIter != MatrVec->end()) {
		*ParIter++ = (void*)(*MatrIter)->CreateMatrix();
		*ParIter++ = (void*)(*MatrIter++)->GetLength();
	}
	return Pars;
}

static void DeleteMatrix(vector<SMatrix*>* MatrVec)
{
	if(MatrVec == nullptr)
		return;
	for(vector<SMatrix*>::iterator MatrIter = MatrVec->begin(); MatrIter != MatrVec->end(); MatrIter++)
		delete *MatrIter;
	delete MatrVec;
}