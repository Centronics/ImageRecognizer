#include <iostream>
#include <boost/filesystem.hpp>
#include "InitGdiPlus.h"
#include <vector>
#include <boost/thread/thread.hpp>
using namespace std;
using namespace boost::filesystem;

wstring BtmExt(L".png");

struct SMatrix
{
	SMatrix(path SN) : SymbolName(SN), Length(), MatrixPtr()
	{
		if(SN.empty())
			throw exception("Путь к символу пуст");
	}

	~SMatrix()
	{
		delete[] MatrixPtr;
	}

	SignType* CreateMatrix(LengthType len = LengthType(), LengthType Width = LengthType())
	{
		if(!len && MatrixPtr != nullptr)
			return MatrixPtr;
		if(!len || !Width)
			throw exception("Длина равна нолю");

		MatrixPtr = new SignType[static_cast<size_t>(len)];
		Length = len;
		return MatrixPtr;
	}

	LengthType GetLength() const
	{
		return Length;
	}

	wstring GetMatrixName() const
	{
		return SymbolName.filename().wstring();
	}

	path GetMatrixPath() const
	{
		return SymbolName;
	}

private:
	SignType* MatrixPtr;
	LengthType Length;
	path SymbolName;
};

bool MatrixCorrect(SMatrix* matr)
{
	register LengthType Length = matr->GetLength() - 1;
	register const SignType* Iter = matr->CreateMatrix() + 1;
	register const SignType* PrevIter = matr->CreateMatrix();
	if(!(*PrevIter))
		return false;
	while(Length--) {
		if(!(*Iter))
			return false;
		if(*PrevIter++ >= *Iter++)
			return false;
	}
	return true;
}

static SignType MinOffSign = SIGNMAX;

void GetMatrix(SMatrix* const smx)
{
	try
	{
		cout << smx->GetMatrixPath().wstring() << endl;
		Bitmap btmPicture(smx->GetMatrixPath().c_str());
		if(btmPicture.GetLastStatus() != Status::Ok)
			return;
		const auto Width  = btmPicture.GetWidth();
		const auto Height = btmPicture.GetHeight();
		const auto sum = Width * Height;
		const SignType SignStep = SIGNMAX / static_cast<SignType>(sum);
		const double DSignStep = static_cast<double>(static_cast<double>(SignStep));
		if(DSignStep < SIGNMIN)
			return;
		SignType* MatrixPtr = smx->CreateMatrix(static_cast<LengthType>(sum), static_cast<LengthType>(Width));
		SignType Sign = SignType(); Color cl;
		for (unsigned int y = 0; y < Height; y++) {
			for (unsigned int x = 0; x < Width; x++)
			{
				btmPicture.GetPixel(x, y, &cl);
				register unsigned long rgb = static_cast<unsigned long>(cl.GetValue());
				rgb &= 0xFFFFFF;
				rgb =  16777215 - rgb;
				double tmpSign = static_cast<double>(rgb) / 16777215.0;
				tmpSign *= DSignStep;
				if(tmpSign < SIGNMIN)
					tmpSign = SIGNMIN;
#if !defined BLOCKALG
				*MatrixPtr++ = Sign += static_cast<SignType>(tmpSign);
#else
				*MatrixPtr++ = Sign + static_cast<SignType>(tmpSign);
				Sign += SignStep;
#endif
			}
		}
		if(MinOffSign > (SIGNMAX - *(MatrixPtr - 1)))
			MinOffSign = SIGNMAX - *(MatrixPtr - 1);
#if defined MATRIXCONTROL
		bool tstres = MatrixCorrect(smx);
#endif
	}
	catch(...) { }
}

bool CmpNoCase(const wstring& s1, const wstring& s2)
{
	if(s1.length() != s2.length())
		return false;
	wstring::const_iterator p1 = s1.begin();
	wstring::const_iterator p2 = s2.begin();
	while(p1 != s1.end() && p2 != s2.end())
		if(toupper(*p1++) != toupper(*p2++))
			return false;
	return true;
}

void ThreadMatrix(vector<SMatrix*>::const_iterator begin, size_t Count)
{
	while(Count--)
		GetMatrix(*begin++);
}

vector<SMatrix*>* GetMatrixes(const path& RootPath)
{
	if(RootPath.empty() || !RootPath.wstring().length())
		return nullptr;
	InitGdiPlus GDI_INIT;
	if(GDI_INIT.Good()) {
		vector<SMatrix*>* MatrVec = new vector<SMatrix*>;
		for (recursive_directory_iterator dir(RootPath), end; dir != end; ++dir) {
			path pth = dir->path();
			if(CmpNoCase(pth.extension().wstring(), BtmExt))
				MatrVec->push_back(new SMatrix(pth));
		}
		if(!MatrVec->size())
			return nullptr;
		const size_t Iters = static_cast<size_t>(MatrVec->size());
			  size_t newThreads = static_cast<size_t>(boost::thread::hardware_concurrency() * 2);
		size_t Count = (newThreads < Iters) ? Iters / newThreads : 1;
		if(Count == 1) {
			Count = Iters;
			newThreads = 0;
		}
		else
			newThreads = Iters / Count;
		boost::thread** const threads = (newThreads ? new boost::thread*[newThreads] : nullptr);
		boost::thread** CurThread = threads;
		vector<SMatrix*>::const_iterator MatrIter = MatrVec->begin();
		for(size_t k = 0; k < Iters; k += Count) {
			if((Iters - k) > Count) {
				*CurThread++ = new boost::thread(&ThreadMatrix, MatrIter + k, Count);
				continue;
			}
			ThreadMatrix(MatrIter + k, Iters - k);
			break;
		}
		size_t tmpThrsNum = CurThread - threads;
		CurThread = threads;
		while(tmpThrsNum--) {
			(*CurThread)->join();
			delete *CurThread++;
		}
		delete[] threads;
		return MatrVec;
	}
	return nullptr;
}