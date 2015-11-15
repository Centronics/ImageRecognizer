#include <windows.h>
#include <gdiplus.h>
using namespace Gdiplus;
#pragma comment (lib, "gdiplus.lib")

#ifndef _DEBUG
#pragma comment(lib, "delayimp.lib")
#pragma comment(linker, "/delayload:gdiplus.dll")
#endif

class InitGdiPlus  
{
public:
	InitGdiPlus();
	~InitGdiPlus();
	bool Good() const { return present; }
private:
	bool present;
	ULONG_PTR token;
};