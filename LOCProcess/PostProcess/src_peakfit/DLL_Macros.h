// Macros to define DLL API entry points
#pragma once

#ifndef DLL_MACROS_H
#define DLL_MACROS_H

#define DLL_CALLCONV __cdecl


#ifdef _WIN32
	#define DLL_EXPORT __declspec(dllexport)
#else
	#define DLL_EXPORT 
#endif


#ifdef __cplusplus
	#define CDLL_EXPORT extern "C" DLL_EXPORT
#else
	#define CDLL_EXPORT DLL_EXPORT
#endif

#endif