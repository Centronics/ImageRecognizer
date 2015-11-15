#include "CUDAMatrix.cuh"
#include "cuda_runtime.h"

const size_t DESCRLEN = 300u;
char ErrorText[DESCRLEN] = "";

SignType* OnCUDA_AllocSignsMem(const size_t NumOfMatrix, const LengthType SignIter)
{
	SignType* bitSigns = 0;
	const size_t conSize = sizeof(SignType) * NumOfMatrix * SignIter;
	cudaError_t cudaStatus = cudaMalloc((void**)&bitSigns, conSize);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMalloc failed!");
		return 0;
	}
	cudaStatus = cudaMemset(bitSigns, 0, conSize);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMemset failed!");
		cudaFree(bitSigns);
		return 0;
	}
	return bitSigns;
}

void FreePagesMemOnCuda(SignType** const Arg, const size_t Count)
{
	const size_t sz = Count * sizeof(SignType*);
	SignType** const PageMas = (SignType**)malloc(sz);
	cudaError_t cudaStatus = cudaMemcpy(PageMas, Arg, sz, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMemcpy failed!");
		return;
	}
	cudaFree(Arg);
	for(size_t k = 0; k < Count; k++)
		cudaFree(PageMas[k]);
	free(PageMas);
}

void FreeRemovedElemsOnCuda(LengthType** Arg)
{
	LengthType* cudaMemory = 0;
	cudaError_t cudaStatus = cudaMemcpy(&cudaMemory, Arg, sizeof(LengthType*), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMemset failed!");
		return;
	}
	cudaFree(cudaMemory);
	cudaFree(Arg);
}

SignType** OnCUDA_AllocPagesMem(void** const ArgPtr, const size_t Count)
{
	const size_t sz = Count * sizeof(SignType*);
	SignType** bitMas = 0;
	cudaError_t cudaStatus = cudaMalloc((void**)&bitMas, sz);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMalloc failed!");
		goto Error;
	}
	SignType** const PageMas = (SignType**)malloc(sz);
	memset(PageMas, 0, sz);
	if(!PageMas)
		goto Error;
	const SignType** ptrMas = (const SignType**)ArgPtr;
	const size_t* ptrSize = (const size_t*)&ArgPtr[1];
	for(size_t k = 0; k < Count; k++)
	{
		cudaStatus = cudaMalloc((void**)&PageMas[k], *ptrSize * sizeof(SignType));
		if (cudaStatus != cudaSuccess) {
			strcpy(ErrorText, "cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(PageMas[k], *ptrMas, *ptrSize * sizeof(SignType), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			strcpy(ErrorText, "cudaMemcpy failed!");
			goto Error;
		}
		ptrMas += 2;
		ptrSize += 2;
	}
	cudaStatus = cudaMemcpy(bitMas, PageMas, sz, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMemcpy failed!");
		goto Error;
	}
	free(PageMas);
	return bitMas;
Error:
	free(PageMas);
	FreePagesMemOnCuda(bitMas, Count);
	return 0;
}

LengthType* OnCUDA_AllocLength(void** const ArgPtr, const size_t Count)
{
	const size_t conSize = sizeof(LengthType) * Count;
	LengthType* const ptrLen = (LengthType*)malloc(conSize);
	if(!ptrLen)
		return 0;
	size_t* CurLenPtr = (size_t*)&ArgPtr[1];
	for(size_t k = 0; k < Count; k++)
	{
		ptrLen[k] = *CurLenPtr;
		CurLenPtr += 2;
	}
	LengthType* LenMas = 0;
	cudaError_t cudaStatus = cudaMalloc((void**)&LenMas, conSize);
	if (cudaStatus != cudaSuccess)
		strcpy(ErrorText, "cudaMalloc failed!");
	cudaStatus = cudaMemcpy(LenMas, ptrLen, conSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		strcpy(ErrorText, "cudaMemcpy failed!");
	free(ptrLen);
	return LenMas;
}

LengthType** OnCUDA_AllocRemoved(const LengthType Count, const LengthType Size, int SetValue)
{
	const size_t AllBytes = Size * Count * sizeof(LengthType), LenBytes = Count * sizeof(LengthType*);
	LengthType** const Lengths = (LengthType**)malloc(LenBytes);
	LengthType* LenMas = 0;
	cudaError_t cudaStatus = cudaMalloc((void**)&LenMas, AllBytes);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMalloc failed!");
		goto Error;
	}
	LengthType** MasPtrLen = 0;
	cudaStatus = cudaMalloc((void**)&MasPtrLen, LenBytes);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(LenMas, SetValue, AllBytes);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMemset failed!");
		goto Error;
	}
	for(LengthType k = 0; k < Count; k++)
		Lengths[k] = &LenMas[k * Size];
	cudaStatus = cudaMemcpy(MasPtrLen, Lengths, LenBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaMemcpy failed!");
		goto Error;
	}
	free(Lengths);
	return MasPtrLen;
Error:
	free(Lengths);
	cudaFree(LenMas);
	cudaFree(MasPtrLen);
	return 0;
}

LengthType* GetTestResults(LengthType* const lens, size_t Count)
{
	Count *= sizeof(LengthType);
	LengthType* const ptrLen = (LengthType*)malloc(Count);
	if(!ptrLen)
		return 0;
	cudaError_t cudaStatus = cudaMemcpy(ptrLen, lens, Count, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		strcpy(ErrorText, "cudaMemcpy failed!");
	return ptrLen;
}

bool WaitForCUDA()
{
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		const char* err = "TestKernel launch failed: ";
		strcpy(ErrorText, err);
		const size_t errLen = strlen(err);
		size_t FreeLen = (DESCRLEN - 1) - errLen;
		const char* cuerr = cudaGetErrorString(cudaStatus);
		const size_t cuerrLen = strlen(cuerr);
		if(FreeLen > cuerrLen)
			FreeLen = cuerrLen;
		memcpy(&ErrorText[errLen], cuerr, FreeLen * sizeof(char));
		ErrorText[errLen + FreeLen] = '\0';
		return false;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		char* err = "cudaDeviceSynchronize returned error code: ";
		size_t errLen = strlen(err);
		strcpy(ErrorText, err);
		_ultoa(cudaStatus, &ErrorText[errLen], 10);
		return false;
	}
	return true;
}

static size_t MasksCount = 0;
static SignType* LoadSignsMaskThread = 0;

LengthType* TestWithCuda(	   void**     const ImageMas,   const size_t Count, const LengthType SignIter,
						 const LengthType       ElemsCount, const bool   Load,  const SignType signAddition)
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus= cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		strcpy(ErrorText, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	if(Load) {
		MasksCount = Count;
		SignType**  const LoadSignsSymbMas	  = OnCUDA_AllocPagesMem(ImageMas, Count);
		SignType**  const LoadCopyPerThread   = OnCUDA_AllocPagesMem(ImageMas, Count);
		LengthType* const MaskLengths		  = OnCUDA_AllocLength  (ImageMas, Count);
						  LoadSignsMaskThread = OnCUDA_AllocSignsMem(Count, SignIter);
		LengthType**const RemovedElems		  = OnCUDA_AllocRemoved (Count, ElemsCount, -1);
		if(signAddition) {
			size_t NeedMem = 0, *ptrSize = (size_t*)ImageMas + 1;
			for(size_t k = 0; k < Count; k++) {
				NeedMem += *ptrSize;
				ptrSize += 2;
			}
			KernelMatrixChange<<<Count, 1>>>(LoadSignsSymbMas, MasksCount, LoadCopyPerThread,
				MaskLengths, NeedMem, signAddition);
		}
		KernelMatrixLoad<<<Count, 1>>>((const SignType** const)LoadSignsSymbMas, LoadCopyPerThread, MaskLengths,
			LoadSignsMaskThread, RemovedElems, ElemsCount, SignIter);
		FreePagesMemOnCuda(LoadSignsSymbMas, Count);
		FreePagesMemOnCuda(LoadCopyPerThread, Count);
		FreeRemovedElemsOnCuda(RemovedElems);
		cudaFree((void*)MaskLengths);
		WaitForCUDA();
		return 0;
	}
	if(!MasksCount || !LoadSignsMaskThread)
		return 0;
	SignType**  const MainSymbMas	  = OnCUDA_AllocPagesMem(ImageMas, Count);
	SignType**  const CopyPerThread   = OnCUDA_AllocPagesMem(ImageMas, Count);
	LengthType* const MainSymbLength  = OnCUDA_AllocLength  (ImageMas, Count);
	SignType*   const SignsPerThread  = OnCUDA_AllocSignsMem(Count, SignIter);
	LengthType**const RemovedElems	  = OnCUDA_AllocRemoved (Count, ElemsCount, -1);
	LengthType**const MatrixTable	  = OnCUDA_AllocRemoved (Count, MasksCount,  0);
	KernelMatrix<<<Count, 1>>>((const SignType** const)MainSymbMas, MasksCount, CopyPerThread, MainSymbLength,
		SignsPerThread, LoadSignsMaskThread, RemovedElems, ElemsCount, SignIter, MatrixTable);
	FreePagesMemOnCuda(MainSymbMas, Count);
	FreePagesMemOnCuda(CopyPerThread, Count);
	FreeRemovedElemsOnCuda(RemovedElems);
	FreeRemovedElemsOnCuda(MatrixTable);
	cudaFree(SignsPerThread);
	LengthType* tmpResults = (WaitForCUDA() ? GetTestResults(MainSymbLength, Count) : 0);
	cudaFree(MainSymbLength);
	return tmpResults;
}