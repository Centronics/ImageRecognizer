#include "InitGdiPlus.h"

InitGdiPlus::InitGdiPlus()
{
	try
	{
		GdiplusStartupInput input;
		GdiplusStartup(&token, &input, 0);
		present = true;
		return;
	}
	catch(...) { }
	present = false;
}

InitGdiPlus::~InitGdiPlus()
{
	if(present)
		Gdiplus::GdiplusShutdown(token);
}