#include "device_launch_parameters.h"

__forceinline__ __device__ SignType MatrixTest(const SignType*   const MainMas,		 SignType* const CopyPerThread,
											   const LengthType		   Length,		 SignType		kSign,
													 LengthType* const RemovedItems, LengthType		Count);

__forceinline__ __device__ void GetSignMassive(const SignType*   const MainSymbMas,		   SignType*   const CopyPerThread,
											   const LengthType		   Length,			   SignType*   const SignsPerThread,
													 LengthType* const RemovedItems, const LengthType		 ElemsCount,
											   const LengthType		   SignStep)
{
	const SignType CurSignStep = SIGNMAX / (SignType)SignStep;
	SignType kSign = 0;
	for(LengthType k = 0; k < SignStep; k++) {
		SignsPerThread[k] = MatrixTest(MainSymbMas, CopyPerThread, Length, kSign, RemovedItems, ElemsCount);
		kSign += CurSignStep;
	}
}

__forceinline__ __device__ void CompareSigns(const SignType*   const MainSymbMas,		    SignType*   const CopyPerThread,
												   LengthType* const MainLength,			SignType*   const SignsPerThread,
												   SignType*   const SignsMaskThread, const LengthType		  MasksLenLength,
												   LengthType* const RemovedItems,    const LengthType		  ElemsCount,
											 const LengthType		 SignIter,				LengthType* const MatrixTable)
{
	GetSignMassive(MainSymbMas, CopyPerThread, *MainLength, SignsPerThread, RemovedItems, ElemsCount, SignIter);
	LengthType p = 0, k = 0; SignType sign; LengthType MatrixNum;
	const LengthType SignIterMasks = MasksLenLength * SignIter;
	for(; k < SignIter; k++) {
		sign = SignsPerThread[k]; MatrixNum = 0;
		SignType Diff = SIGNMAX, psign = SIGNMAX;
		for(p = 0; p < SignIterMasks; p += SignIter) {
			psign = (SignsMaskThread + p)[k];
			const SignType diff = (sign > psign) ? sign - psign : psign - sign;
			if(!diff) {
				MatrixTable[p / SignIter]++;
				MatrixNum = LENGTHMAX;
			}
			if(diff < Diff && MatrixNum != LENGTHMAX) {
				Diff = diff;
				MatrixNum = p / SignIter;
			}
		}
		if(MatrixNum != LENGTHMAX)
			MatrixTable[MatrixNum]++;
	}
	p = 0; MatrixNum = 0;
	for(k = 0; k < MasksLenLength; k++)
		if(MatrixTable[k] > p) {
			p = MatrixTable[k];
			MatrixNum = k;
		}
	*MainLength = MatrixNum;
}

__global__ void KernelMatrixLoad(const SignType**   const SignsSymbMas,			SignType**   const CopyPerThread,
								 const LengthType*	const MainSymbLength,		SignType*    const SignsMaskThread,
								       LengthType** const RemovedItems,   const LengthType		   ElemsCount,
								 const LengthType		  SignIter)
{
	unsigned int Me = threadIdx.x + (blockIdx.x * blockDim.x);
	GetSignMassive(SignsSymbMas[Me], CopyPerThread[Me], MainSymbLength[Me], &SignsMaskThread[Me * SignIter],
		RemovedItems[Me], ElemsCount, SignIter);
}

__global__ void KernelMatrix(const SignType**   const MainSymbMas,	  const LengthType		  MasksLenLength,
							 	   SignType**   const CopyPerThread,		LengthType*	const CopyLength,
								   SignType*    const SignsPerThread,       SignType*   const SignsMaskThread,
								   LengthType** const RemovedItems,   const LengthType		  ElemsCount,
							 const LengthType		  SignIter,				LengthType**const MatrixTable)
{
	unsigned int Me = threadIdx.x + (blockIdx.x * blockDim.x);
	CompareSigns(MainSymbMas[Me], CopyPerThread[Me], &CopyLength[Me], &SignsPerThread[Me * SignIter],
		SignsMaskThread, MasksLenLength, RemovedItems[Me], ElemsCount, SignIter, MatrixTable[Me]);
}

__global__ void KernelMatrixChange(		 SignType** const MainSymbMas,  const LengthType	MasksLenLength,
							 			 SignType** const CopyPerThread,	  LengthType*	CopyLength,  
								   const size_t			  AllSumm,		const SignType		Addition)
{
	unsigned int Me = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int MyCount = AllSumm / gridDim.x, ost = AllSumm % gridDim.x;
	if(Me < ost)
		MyCount++;
	LengthType Start = Me;
	SignType *MyPointer = *MainSymbMas, *EndPointer = 0;
	SignType *MyCopyPointer = *CopyPerThread;
	for(LengthType p = 0; p < MasksLenLength; p++) {
		if(Start >= CopyLength[p])
			Start -= CopyLength[p];
		else {
			MyPointer = &MainSymbMas[p][Start];
			MyCopyPointer = &CopyPerThread[p][Start];
			if((Start + MyCount) >= CopyLength[p])
				EndPointer = &MainSymbMas[p][CopyLength[p]];
			else
				EndPointer = &MyPointer[MyCount];
			break;
		}
	}
	__syncthreads();
	while(MyPointer != EndPointer) {
		*MyPointer += Addition;
		*MyCopyPointer++ = *MyPointer++;
	}
}


__forceinline__ __device__ LengthType FindNextSign(SignType* const FindMas, LengthType Number, LengthType Length, bool UP)
{
	if(UP) {
		for( ; Number < Length; Number++)
			if(FindMas[Number])
				return Number;
		return LENGTHMAX;
	}
	do {
		if(FindMas[Number])
			return Number;
	}
	while(Number--);
	return LENGTHMAX;
}

__forceinline__ __device__ SignType FindAndRemove(const SignType   Sign,   SignType*   const FindMas,
												  const LengthType Length, LengthType* const RemovedItem)
{
	LengthType NumberMin = 0, NumberMax = Length - 1;
	LengthType RetNumber = 0;
	while(NumberMin < NumberMax)
	{
		LengthType Number = NumberMax - NumberMin;
		Number /= 2;
		Number += NumberMin;
		LengthType mNumber1 = FindNextSign(FindMas, Number, Length, false);
		LengthType mNumber2 = FindNextSign(FindMas, Number + 1, Length, true);
		if(mNumber1 == LENGTHMAX || mNumber2 == LENGTHMAX) {
			if(mNumber1 != LENGTHMAX)
				RetNumber = mNumber1;
			else
				if(mNumber2 != LENGTHMAX)
					RetNumber = mNumber2;
				else
					return 0;
			break;
		}
		SignType ds1 = FindMas[mNumber1];
				 ds1 = (ds1 < Sign) ? Sign - ds1 : ds1 - Sign;
		SignType ds2 = FindMas[mNumber2];
				 ds2 = (ds2 < Sign) ? Sign - ds2 : ds2 - Sign;
		if(ds1 > ds2) {
			NumberMin = Number += 1;
			RetNumber = mNumber2;
		}
		else {
			NumberMax = Number;
			RetNumber = mNumber1;
		}
	}
	SignType tmp = FindMas[RetNumber];
	FindMas[RetNumber] = 0;
	*RemovedItem = RetNumber;
	return tmp;
}

__forceinline__ __device__ SignType MatrixTest(const SignType*   const MainMas,			   SignType* const CopyPerThread,
											   const LengthType		   Length,			   SignType		   kSign,
													 LengthType* const RemovedItems, const LengthType	   Count)
{
	SignType FindSign = FindAndRemove(kSign, CopyPerThread, Length, RemovedItems);
	if (!FindSign)
		return 0;
	kSign = (kSign / 2) + (FindSign / 2);
#if !defined FLOATSIGN
	kSign += (kSign % 2) & (FindSign % 2);
#endif
	LengthType NCount = Count;
	while (FindSign && --NCount) {
		FindSign = FindAndRemove(kSign, CopyPerThread, Length, &RemovedItems[NCount]);
		kSign = (kSign / 2) + (FindSign / 2);
#if !defined FLOATSIGN
		kSign += (kSign % 2) & (FindSign % 2);
#endif
	}
	for(LengthType k = 0; k < Count; k++) {
		const LengthType t = RemovedItems[k];
		CopyPerThread[t] = MainMas[t];
	}
	return kSign;
}