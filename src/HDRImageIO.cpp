//
// Copyright (C) Wojciech Jarosz <wjarosz@gmail.com>. All rights reserved.
// Use of this source code is governed by a BSD-style license that can
// be found in the LICENSE.txt file.
//

#include "HDRImage.h"
#include "DitherMatrix256.h"    // for dither_matrix256
#include <ImfArray.h>            // for Array2D
#include <ImfRgbaFile.h>         // for RgbaInputFile, RgbaOutputFile
#include <ImathBox.h>            // for Box2i
#include <ImfTestFile.h>         // for isOpenExrFile
#include <ImathVec.h>            // for Vec2
#include <ImfRgba.h>             // for Rgba, RgbaChannels::WRITE_RGBA
#include <ctype.h>               // for tolower
#include <half.h>                // for half
#include <stdlib.h>              // for abs
#include <algorithm>             // for nth_element, transform
#include <cmath>                 // for floor, pow, exp, ceil, round, sqrt
#include <exception>             // for exception
#include <functional>            // for pointer_to_unary_function, function
#include <stdexcept>             // for runtime_error, out_of_range
#include <string>                // for allocator, operator==, basic_string
#include <vector>                // for vector
#include "Common.h"              // for lerp, mod, clamp, getExtension
#include "Colorspace.h"
#include "ParallelFor.h"
#include "Timer.h"
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <TinyNPY.h>

// these pragmas ignore warnings about unused static functions
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#elif defined(_MSC_VER)
#pragma warning (push, 0)
#endif

// since NanoVG includes an old version of stb_image, we declare it static here
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#define TINY_DNG_LOADER_IMPLEMENTATION
#include "tiny_dng_loader.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning (pop)
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"     // for stbi_write_bmp, stbi_write_hdr, stbi...

#include "PFM.h"
#include "PPM.h"


using namespace Eigen;
using namespace std;

// local functions
namespace
{

inline unsigned short endianSwap(unsigned short val);
void decode12BitToFloat(vector<float> &image, unsigned char *data, int width, int height, bool swapEndian);
void decode14BitToFloat(vector<float> &image, unsigned char *data, int width, int height, bool swapEndian);
void decode16BitToFloat(vector<float> &image, unsigned char *data, int width, int height, bool swapEndian);
void printImageInfo(const tinydng::DNGImage & image);
HDRImage develop(vector<float> & raw,
                 const tinydng::DNGImage & param1,
                 const tinydng::DNGImage & param2);
void copyPixelsFromArray(HDRImage & img, float * data, int w, int h, int n, bool convertToLinear, bool flip)
{
	if (n != 1 && n != 3 && n != 4)
		throw runtime_error("Only 1- 3- and 4-channel images are supported.");

	// for every pixel in the image
	if (n == 1) {
		#if 1
		static const float turboRGBf[256][3] = {{0.18995f,0.07176f,0.23217f},{0.19483f,0.08339f,0.26149f},{0.19956f,0.09498f,0.29024f},{0.20415f,0.10652f,0.31844f},{0.20860f,0.11802f,0.34607f},{0.21291f,0.12947f,0.37314f},{0.21708f,0.14087f,0.39964f},{0.22111f,0.15223f,0.42558f},{0.22500f,0.16354f,0.45096f},{0.22875f,0.17481f,0.47578f},{0.23236f,0.18603f,0.50004f},{0.23582f,0.19720f,0.52373f},{0.23915f,0.20833f,0.54686f},{0.24234f,0.21941f,0.56942f},{0.24539f,0.23044f,0.59142f},{0.24830f,0.24143f,0.61286f},{0.25107f,0.25237f,0.63374f},{0.25369f,0.26327f,0.65406f},{0.25618f,0.27412f,0.67381f},{0.25853f,0.28492f,0.69300f},{0.26074f,0.29568f,0.71162f},{0.26280f,0.30639f,0.72968f},{0.26473f,0.31706f,0.74718f},{0.26652f,0.32768f,0.76412f},{0.26816f,0.33825f,0.78050f},{0.26967f,0.34878f,0.79631f},{0.27103f,0.35926f,0.81156f},{0.27226f,0.36970f,0.82624f},{0.27334f,0.38008f,0.84037f},{0.27429f,0.39043f,0.85393f},{0.27509f,0.40072f,0.86692f},{0.27576f,0.41097f,0.87936f},{0.27628f,0.42118f,0.89123f},{0.27667f,0.43134f,0.90254f},{0.27691f,0.44145f,0.91328f},{0.27701f,0.45152f,0.92347f},{0.27698f,0.46153f,0.93309f},{0.27680f,0.47151f,0.94214f},{0.27648f,0.48144f,0.95064f},{0.27603f,0.49132f,0.95857f},{0.27543f,0.50115f,0.96594f},{0.27469f,0.51094f,0.97275f},{0.27381f,0.52069f,0.97899f},{0.27273f,0.53040f,0.98461f},{0.27106f,0.54015f,0.98930f},{0.26878f,0.54995f,0.99303f},{0.26592f,0.55979f,0.99583f},{0.26252f,0.56967f,0.99773f},{0.25862f,0.57958f,0.99876f},{0.25425f,0.58950f,0.99896f},{0.24946f,0.59943f,0.99835f},{0.24427f,0.60937f,0.99697f},{0.23874f,0.61931f,0.99485f},{0.23288f,0.62923f,0.99202f},{0.22676f,0.63913f,0.98851f},{0.22039f,0.64901f,0.98436f},{0.21382f,0.65886f,0.97959f},{0.20708f,0.66866f,0.97423f},{0.20021f,0.67842f,0.96833f},{0.19326f,0.68812f,0.96190f},{0.18625f,0.69775f,0.95498f},{0.17923f,0.70732f,0.94761f},{0.17223f,0.71680f,0.93981f},{0.16529f,0.72620f,0.93161f},{0.15844f,0.73551f,0.92305f},{0.15173f,0.74472f,0.91416f},{0.14519f,0.75381f,0.90496f},{0.13886f,0.76279f,0.89550f},{0.13278f,0.77165f,0.88580f},{0.12698f,0.78037f,0.87590f},{0.12151f,0.78896f,0.86581f},{0.11639f,0.79740f,0.85559f},{0.11167f,0.80569f,0.84525f},{0.10738f,0.81381f,0.83484f},{0.10357f,0.82177f,0.82437f},{0.10026f,0.82955f,0.81389f},{0.09750f,0.83714f,0.80342f},{0.09532f,0.84455f,0.79299f},{0.09377f,0.85175f,0.78264f},{0.09287f,0.85875f,0.77240f},{0.09267f,0.86554f,0.76230f},{0.09320f,0.87211f,0.75237f},{0.09451f,0.87844f,0.74265f},{0.09662f,0.88454f,0.73316f},{0.09958f,0.89040f,0.72393f},{0.10342f,0.89600f,0.71500f},{0.10815f,0.90142f,0.70599f},{0.11374f,0.90673f,0.69651f},{0.12014f,0.91193f,0.68660f},{0.12733f,0.91701f,0.67627f},{0.13526f,0.92197f,0.66556f},{0.14391f,0.92680f,0.65448f},{0.15323f,0.93151f,0.64308f},{0.16319f,0.93609f,0.63137f},{0.17377f,0.94053f,0.61938f},{0.18491f,0.94484f,0.60713f},{0.19659f,0.94901f,0.59466f},{0.20877f,0.95304f,0.58199f},{0.22142f,0.95692f,0.56914f},{0.23449f,0.96065f,0.55614f},{0.24797f,0.96423f,0.54303f},{0.26180f,0.96765f,0.52981f},{0.27597f,0.97092f,0.51653f},{0.29042f,0.97403f,0.50321f},{0.30513f,0.97697f,0.48987f},{0.32006f,0.97974f,0.47654f},{0.33517f,0.98234f,0.46325f},{0.35043f,0.98477f,0.45002f},{0.36581f,0.98702f,0.43688f},{0.38127f,0.98909f,0.42386f},{0.39678f,0.99098f,0.41098f},{0.41229f,0.99268f,0.39826f},{0.42778f,0.99419f,0.38575f},{0.44321f,0.99551f,0.37345f},{0.45854f,0.99663f,0.36140f},{0.47375f,0.99755f,0.34963f},{0.48879f,0.99828f,0.33816f},{0.50362f,0.99879f,0.32701f},{0.51822f,0.99910f,0.31622f},{0.53255f,0.99919f,0.30581f},{0.54658f,0.99907f,0.29581f},{0.56026f,0.99873f,0.28623f},{0.57357f,0.99817f,0.27712f},{0.58646f,0.99739f,0.26849f},{0.59891f,0.99638f,0.26038f},{0.61088f,0.99514f,0.25280f},{0.62233f,0.99366f,0.24579f},{0.63323f,0.99195f,0.23937f},{0.64362f,0.98999f,0.23356f},{0.65394f,0.98775f,0.22835f},{0.66428f,0.98524f,0.22370f},{0.67462f,0.98246f,0.21960f},{0.68494f,0.97941f,0.21602f},{0.69525f,0.97610f,0.21294f},{0.70553f,0.97255f,0.21032f},{0.71577f,0.96875f,0.20815f},{0.72596f,0.96470f,0.20640f},{0.73610f,0.96043f,0.20504f},{0.74617f,0.95593f,0.20406f},{0.75617f,0.95121f,0.20343f},{0.76608f,0.94627f,0.20311f},{0.77591f,0.94113f,0.20310f},{0.78563f,0.93579f,0.20336f},{0.79524f,0.93025f,0.20386f},{0.80473f,0.92452f,0.20459f},{0.81410f,0.91861f,0.20552f},{0.82333f,0.91253f,0.20663f},{0.83241f,0.90627f,0.20788f},{0.84133f,0.89986f,0.20926f},{0.85010f,0.89328f,0.21074f},{0.85868f,0.88655f,0.21230f},{0.86709f,0.87968f,0.21391f},{0.87530f,0.87267f,0.21555f},{0.88331f,0.86553f,0.21719f},{0.89112f,0.85826f,0.21880f},{0.89870f,0.85087f,0.22038f},{0.90605f,0.84337f,0.22188f},{0.91317f,0.83576f,0.22328f},{0.92004f,0.82806f,0.22456f},{0.92666f,0.82025f,0.22570f},{0.93301f,0.81236f,0.22667f},{0.93909f,0.80439f,0.22744f},{0.94489f,0.79634f,0.22800f},{0.95039f,0.78823f,0.22831f},{0.95560f,0.78005f,0.22836f},{0.96049f,0.77181f,0.22811f},{0.96507f,0.76352f,0.22754f},{0.96931f,0.75519f,0.22663f},{0.97323f,0.74682f,0.22536f},{0.97679f,0.73842f,0.22369f},{0.98000f,0.73000f,0.22161f},{0.98289f,0.72140f,0.21918f},{0.98549f,0.71250f,0.21650f},{0.98781f,0.70330f,0.21358f},{0.98986f,0.69382f,0.21043f},{0.99163f,0.68408f,0.20706f},{0.99314f,0.67408f,0.20348f},{0.99438f,0.66386f,0.19971f},{0.99535f,0.65341f,0.19577f},{0.99607f,0.64277f,0.19165f},{0.99654f,0.63193f,0.18738f},{0.99675f,0.62093f,0.18297f},{0.99672f,0.60977f,0.17842f},{0.99644f,0.59846f,0.17376f},{0.99593f,0.58703f,0.16899f},{0.99517f,0.57549f,0.16412f},{0.99419f,0.56386f,0.15918f},{0.99297f,0.55214f,0.15417f},{0.99153f,0.54036f,0.14910f},{0.98987f,0.52854f,0.14398f},{0.98799f,0.51667f,0.13883f},{0.98590f,0.50479f,0.13367f},{0.98360f,0.49291f,0.12849f},{0.98108f,0.48104f,0.12332f},{0.97837f,0.46920f,0.11817f},{0.97545f,0.45740f,0.11305f},{0.97234f,0.44565f,0.10797f},{0.96904f,0.43399f,0.10294f},{0.96555f,0.42241f,0.09798f},{0.96187f,0.41093f,0.09310f},{0.95801f,0.39958f,0.08831f},{0.95398f,0.38836f,0.08362f},{0.94977f,0.37729f,0.07905f},{0.94538f,0.36638f,0.07461f},{0.94084f,0.35566f,0.07031f},{0.93612f,0.34513f,0.06616f},{0.93125f,0.33482f,0.06218f},{0.92623f,0.32473f,0.05837f},{0.92105f,0.31489f,0.05475f},{0.91572f,0.30530f,0.05134f},{0.91024f,0.29599f,0.04814f},{0.90463f,0.28696f,0.04516f},{0.89888f,0.27824f,0.04243f},{0.89298f,0.26981f,0.03993f},{0.88691f,0.26152f,0.03753f},{0.88066f,0.25334f,0.03521f},{0.87422f,0.24526f,0.03297f},{0.86760f,0.23730f,0.03082f},{0.86079f,0.22945f,0.02875f},{0.85380f,0.22170f,0.02677f},{0.84662f,0.21407f,0.02487f},{0.83926f,0.20654f,0.02305f},{0.83172f,0.19912f,0.02131f},{0.82399f,0.19182f,0.01966f},{0.81608f,0.18462f,0.01809f},{0.80799f,0.17753f,0.01660f},{0.79971f,0.17055f,0.01520f},{0.79125f,0.16368f,0.01387f},{0.78260f,0.15693f,0.01264f},{0.77377f,0.15028f,0.01148f},{0.76476f,0.14374f,0.01041f},{0.75556f,0.13731f,0.00942f},{0.74617f,0.13098f,0.00851f},{0.73661f,0.12477f,0.00769f},{0.72686f,0.11867f,0.00695f},{0.71692f,0.11268f,0.00629f},{0.70680f,0.10680f,0.00571f},{0.69650f,0.10102f,0.00522f},{0.68602f,0.09536f,0.00481f},{0.67535f,0.08980f,0.00449f},{0.66449f,0.08436f,0.00424f},{0.65345f,0.07902f,0.00408f},{0.64223f,0.07380f,0.00401f},{0.63082f,0.06868f,0.00401f},{0.61923f,0.06367f,0.00410f},{0.60746f,0.05878f,0.00427f},{0.59550f,0.05399f,0.00453f},{0.58336f,0.04931f,0.00486f},{0.57103f,0.04474f,0.00529f},{0.55852f,0.04028f,0.00579f},{0.54583f,0.03593f,0.00638f},{0.53295f,0.03169f,0.00705f},{0.51989f,0.02756f,0.00780f},{0.50664f,0.02354f,0.00863f},{0.49321f,0.01963f,0.00955f},{0.47960f,0.01583f,0.01055f}};
		float minVal(FLT_MAX), maxVal(-FLT_MAX);
		for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x)
		{
			const float p(data[x + y * w]);
			if (p <= 0)
				continue;
			if (minVal > p)
				minVal = p;
			if (maxVal < p)
				maxVal = p;
		}
		const float delta(maxVal-minVal);
		parallel_for(0, h, [&img,w,h,n,data,convertToLinear,flip,minVal,delta](int y)
		{
			for (int x = 0; x < w; ++x)
			{
				const float p(data[x + y * w]);
				if (p <= 0) {
					img(x, flip?h-y-1:y) = Color4(0,0,0,0);
					continue;
				}
				const float* const t(turboRGBf[(int)std::round(255.f*(p-minVal)/delta)]);
				Color4 c(t[2],
						 t[1],
						 t[0],
						 1.f);
				img(x, flip?h-y-1:y) = SRGBToLinear(c);
			}
		});
		#else
		parallel_for(0, h, [&img,w,data,convertToLinear](int y)
		{
			for (int x = 0; x < w; ++x)
			{
				const float v(data[x + y * w]);
				const Color4 c(v,v,v,1.f);
				img(x, y) = convertToLinear ? SRGBToLinear(c) : c;
			}
		});
		#endif
	} else {
		parallel_for(0, h, [&img,w,h,n,data,convertToLinear,flip](int y)
		{
			for (int x = 0; x < w; ++x)
			{
				Color4 c(data[n * (x + y * w) + 0],
						 data[n * (x + y * w) + 1],
						 data[n * (x + y * w) + 2],
						 (n == 3) ? 1.f : data[4 * (x + y * w) + 3]);
				img(x, flip?h-y-1:y) = convertToLinear ? SRGBToLinear(c) : c;
			}
		});
	}
}

bool isSTBImage(const string & filename)
{
	FILE *f = stbi__fopen(filename.c_str(), "rb");
	if (!f)
		return false;

	stbi__context s;
	stbi__start_file(&s,f);

	// try stb library first
	if (stbi__jpeg_test(&s) ||
		stbi__png_test(&s) ||
		stbi__bmp_test(&s) ||
		stbi__gif_test(&s) ||
		stbi__psd_test(&s) ||
		stbi__pic_test(&s) ||
		stbi__pnm_test(&s) ||
		stbi__hdr_test(&s) ||
		stbi__tga_test(&s))
	{
		fclose(f);
		return true;
	}

	fclose(f);
	return false;
}

} // namespace


bool HDRImage::load(const string & filename)
{
	auto console = spdlog::get("console");
    string errors;
	string extension = getExtension(filename);
	transform(extension.begin(),
	          extension.end(),
	          extension.begin(),
	          ::tolower);

    int n, w, h;

	// try stb library first
	if (isSTBImage(filename))
	{
		// stbi doesn't do proper srgb, but uses gamma=2.2 instead, so override it.
		// we'll do our own srgb correction
		stbi_ldr_to_hdr_scale(1.0f);
		stbi_ldr_to_hdr_gamma(1.0f);

		float * float_data = stbi_loadf(filename.c_str(), &w, &h, &n, 4);
		if (n == 1 && filename.size()>5 && filename.substr(filename.size()-4)==".png") {
			FILE *f = stbi__fopen(filename.c_str(), "rb");
			stbi__context s;
			stbi__start_file(&s,f);
			stbi__result_info ri;
			void *result = stbi__load_main(&s, &w, &h, &n, 1, &ri, 8);
			fclose(f);
			if (ri.bits_per_channel == 16) {
				stbi_image_free(float_data);
				resize(w, h);

				Timer timer;
				// convert 1-channel PNG data to 4-channel internal representation
				const uint16_t* data16 = (const uint16_t*)result;
				std::vector<float> data(w*h);
				for (int i=0; i<w*h; ++i)
					if ((data[i] = ((float)data16[i]) / 1000.f) > 5.f)
						data[i] = 0.f;
				copyPixelsFromArray(*this, data.data(), w, h, n, false, true);
				console->debug("Copying image data took: {} seconds.", (timer.elapsed() / 1000.f));

				return true;
			}
		}
		if (float_data)
		{
			resize(w, h);
			bool convertToLinear = !stbi_is_hdr(filename.c_str());
			Timer timer;
			copyPixelsFromArray(*this, float_data, w, h, 4, convertToLinear, false);
			console->debug("Copying image data took: {} seconds.", (timer.elapsed()/1000.f));

			stbi_image_free(float_data);
			return true;
		}
		else
		{
			errors += string("\t") + stbi_failure_reason() + "\n";
		}
	}


    // then try pfm
	if (isPFMImage(filename.c_str()))
    {
	    float * float_data = 0;
	    try
	    {
		    w = 0;
		    h = 0;

		    if ((float_data = loadPFMImage(filename.c_str(), &w, &h, &n)))
		    {
			    if (n == 3 || n == 1)
			    {
				    resize(w, h);

				    Timer timer;
				    // convert 1- 3-channel pfm data to 4-channel internal representation
				    copyPixelsFromArray(*this, float_data, w, h, n, false, true);
				    console->debug("Copying image data took: {} seconds.", (timer.elapsed() / 1000.f));

				    delete [] float_data;
				    return true;
			    }
			    else
				    throw runtime_error("Only 3-channel or 1-channel PFMs are currently supported.");
			    return true;
		    }
		    else
			    throw runtime_error("Could not load PFM image.");
	    }
	    catch (const exception &e)
	    {
		    delete [] float_data;
		    resize(0, 0);
		    errors += string("\t") + e.what() + "\n";
	    }
    }


    // then try npy
	if (filename.size()>5 && filename.substr(filename.size()-4)==".npy")
    {
		try
	    {
			NpyArray arr;
			if (arr.LoadNPY(filename.c_str()) != NULL)
				throw runtime_error("Could not load NPY image.");
			if (arr.Shape().size() < 2 || arr.Shape().size() > 3)
				throw runtime_error("NPY not an image.");
		    w = arr.Shape()[1];
		    h = arr.Shape()[0];
			n = arr.Shape().size() == 2 ? 1 : arr.Shape()[2];
			if ((n == 3 || n == 4 || n == 1) && arr.ValueType() == typeid(float))
			{
				resize(w, h);

				Timer timer;
				// convert 1- 3-channel NPY data to 4-channel internal representation
				copyPixelsFromArray(*this, (float*)arr.Data(), w, h, n, false, false);
				console->debug("Copying image data took: {} seconds.", (timer.elapsed() / 1000.f));

				return true;
			}
			else
			if (n != 4 || arr.ValueType() != typeid(uint8_t))
				throw runtime_error("Only 1- 3- 4-channel float NPYs are currently supported.");
			return true;
	    }
	    catch (const exception &e)
	    {
		    resize(0, 0);
		    errors += string("\t") + e.what() + "\n";
	    }
    }

    // next try exrs
	if (Imf::isOpenExrFile(filename.c_str()))
    {
	    try
	    {
		    // FIXME: the threading below seems to cause issues, but shouldn't.
		    // turning off for now
		    Imf::setGlobalThreadCount(thread::hardware_concurrency());
		    Timer timer;

		    Imf::RgbaInputFile file(filename.c_str());
		    Imath::Box2i dw = file.dataWindow();

		    w = dw.max.x - dw.min.x + 1;
		    h = dw.max.y - dw.min.y + 1;

		    Imf::Array2D<Imf::Rgba> pixels(h, w);

		    file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * w, 1, w);
		    file.readPixels(dw.min.y, dw.max.y);

		    console->debug("Reading EXR image took: {} seconds.", (timer.lap() / 1000.f));

		    resize(w, h);

		    // copy pixels over to the Image
		    parallel_for(0, h, [this, w, &pixels](int y)
		    {
			    for (int x = 0; x < w; ++x)
			    {
				    const Imf::Rgba &p = pixels[y][x];
				    (*this)(x, y) = Color4(p.r, p.g, p.b, p.a);
			    }
		    });

		    console->debug("Copying EXR image data took: {} seconds.", (timer.lap() / 1000.f));
		    return true;
	    }
	    catch (const exception &e)
	    {
		    resize(0, 0);
		    errors += string("\t") + e.what() + "\n";
	    }
    }

	try
	{
		vector<tinydng::DNGImage> images;
		{
			std::string err;
			vector<tinydng::FieldInfo> customFields;
			bool ret = tinydng::LoadDNG(filename.c_str(), customFields, &images, &err);

			if (ret == false)
				throw runtime_error("Failed to load DNG. " + err);
		}

		// DNG files sometimes only store the orientation in one of the images,
		// instead of all of them. find any set value and save it
		int orientation = 0;
		for (size_t i = 0; i < images.size(); i++)
		{
			console->debug("Image [{}] size = {} x {}.", i, images[i].width, images[i].height);
			console->debug("Image [{}] orientation = {}", i, images[i].orientation);
			if (images[i].orientation != 0)
				orientation = images[i].orientation;
		}

		// Find largest image based on width.
		size_t imageIndex = size_t(-1);
		{
			size_t largest = 0;
			int largestWidth = images[0].width;
			for (size_t i = 0; i < images.size(); i++)
			{
				if (largestWidth < images[i].width)
				{
					largest = i;
					largestWidth = images[i].width;
				}
			}

			imageIndex = largest;
		}
		tinydng::DNGImage & image = images[imageIndex];


		console->debug("\nLargest image within DNG:");
		printImageInfo(image);
		console->debug("\nLast image within DNG:");
		printImageInfo(images.back());

		console->debug("Loading image [{}].", imageIndex);

		w = image.width;
		h = image.height;

		// Convert to float.
		vector<float> hdr;
		bool endianSwap = false;        // TODO

		int spp = image.samples_per_pixel;
		if (image.bits_per_sample == 12)
			decode12BitToFloat(hdr, &(image.data.at(0)), w, h * spp, endianSwap);
		else if (image.bits_per_sample == 14)
			decode14BitToFloat(hdr, &(image.data.at(0)), w, h * spp, endianSwap);
		else if (image.bits_per_sample == 16)
			decode16BitToFloat(hdr, &(image.data.at(0)), w, h * spp, endianSwap);
		else
			throw runtime_error("Error loading DNG: Unsupported bits_per_sample : " + to_string(spp));

		float invScale = 1.0f / static_cast<float>((1 << image.bits_per_sample));
		if (spp == 3)
		{
			console->debug("Decoding a 3 sample-per-pixel DNG image.");
			// normalize
			parallel_for(0, hdr.size(), [&hdr,invScale](int i)
			{
				hdr[i] *= invScale;
			});

			// Create color image & normalize intensity.
			resize(w, h);

			Timer timer;
			// normalize
			parallel_for(0, h, [this,w,invScale,&hdr](int y)
			{
				for (int x = 0; x < w; ++x)
				{
					int index = 3 * y * w + x;
					(*this)(x, y) = Color4(hdr[index] * invScale + 0,
					                       hdr[index] * invScale + 1,
					                       hdr[index] * invScale + 2, 1.0f);
				}
			});
			console->debug("Copying image data took: {} seconds.", (timer.elapsed()/1000.f));
		}
		else if (spp == 1)
		{
			// Create grayscale image & normalize intensity.
			console->debug("Decoding a 1 sample-per-pixel DNG image.");
			Timer timer;
			*this = develop(hdr, image, images.back());
			console->debug("Copying image data took: {} seconds.", (timer.elapsed()/1000.f));
		}
		else
			throw runtime_error("Error loading DNG: Unsupported samples per pixel: " + to_string(spp));


		int startRow = clamp(image.active_area[1], 0, w);
		int endRow = clamp(image.active_area[3], 0, w);
		int startCol = clamp(image.active_area[0], 0, h);
		int endCol = clamp(image.active_area[2], 0, h);

		*this = block(startRow, startCol,
		              endRow-startRow,
		              endCol-startCol).eval();

		enum Orientations
		{
			ORIENTATION_TOPLEFT = 1,
			ORIENTATION_TOPRIGHT = 2,
			ORIENTATION_BOTRIGHT = 3,
			ORIENTATION_BOTLEFT = 4,
			ORIENTATION_LEFTTOP = 5,
			ORIENTATION_RIGHTTOP = 6,
			ORIENTATION_RIGHTBOT = 7,
			ORIENTATION_LEFTBOT = 8
		};

		// now rotate image based on stored orientation
		switch (orientation)
		{
			case ORIENTATION_TOPRIGHT: *this = flippedHorizontal(); break;
			case ORIENTATION_BOTRIGHT: *this = flippedVertical().flippedHorizontal(); break;
			case ORIENTATION_BOTLEFT : *this = flippedVertical(); break;
			case ORIENTATION_LEFTTOP : *this = rotated90CCW().flippedVertical(); break;
			case ORIENTATION_RIGHTTOP: *this = rotated90CW(); break;
			case ORIENTATION_RIGHTBOT: *this = rotated90CW().flippedVertical(); break;
			case ORIENTATION_LEFTBOT : *this = rotated90CCW(); break;
			default: break;// none (0), or ORIENTATION_TOPLEFT
		}

		return true;
	}
	catch (const exception &e)
	{
		resize(0,0);
		// only report errors to the user if the extension was actually dng
		if (extension == "dng")
			errors += string("\t") + e.what() + "\n";
	}

    console->error("ERROR: Unable to read image file \"{}\":\n{}", filename, errors);

    return false;
}


shared_ptr<HDRImage> loadImage(const string & filename)
{
	shared_ptr<HDRImage> ret = make_shared<HDRImage>();
	if (ret->load(filename))
		return ret;
	return nullptr;
}


bool HDRImage::save(const string & filename,
                    float gain, float gamma,
                    bool sRGB, bool dither) const
{
	auto console = spdlog::get("console");
    string extension = getExtension(filename);

    transform(extension.begin(),
              extension.end(),
              extension.begin(),
              ::tolower);

    auto img = this;
    HDRImage imgCopy;

    bool hdrFormat = (extension == "hdr") || (extension == "pfm") || (extension == "exr");

    // if we need to tonemap, then modify a copy of the image data
    if (gain != 1.0f || sRGB || gamma != 1.0f)
    {
        Color4 gainC = Color4(gain, gain, gain, 1.0f);
        Color4 gammaC = Color4(1.0f / gamma, 1.0f / gamma, 1.0f / gamma, 1.0f);

        imgCopy = *this;
        img = &imgCopy;

        if (gain != 1.0f)
            imgCopy *= gainC;

        // only do gamma or sRGB tonemapping if we are saving to an LDR format
        if (!hdrFormat)
        {
            if (sRGB)
                imgCopy = imgCopy.unaryExpr(ptr_fun((Color4 (*)(const Color4 &)) LinearToSRGB));
            else if (gamma != 1.0f)
                imgCopy = imgCopy.pow(gammaC);
        }
    }

    if (extension == "hdr")
        return stbi_write_hdr(filename.c_str(), width(), height(), 4, (const float *) img->data()) != 0;
    else if (extension == "pfm")
        return writePFMImage(filename.c_str(), width(), height(), 4, (const float *) img->data()) != 0;
    else if (extension == "exr")
    {
        try
        {
            Imf::setGlobalThreadCount(thread::hardware_concurrency());
            Imf::RgbaOutputFile file(filename.c_str(), width(), height(), Imf::WRITE_RGBA);
            Imf::Array2D<Imf::Rgba> pixels(height(), width());

            Timer timer;
            // copy image data over to Rgba pixels
            parallel_for(0, height(), [this,img,&pixels](int y)
            {
                for (int x = 0; x < width(); ++x)
                {
                    Imf::Rgba &p = pixels[y][x];
                    Color4 c = (*img)(x, y);
                    p.r = c[0];
                    p.g = c[1];
                    p.b = c[2];
                    p.a = c[3];
                }
            });
            console->debug("Copying pixel data took: {} seconds.", (timer.lap()/1000.f));

            file.setFrameBuffer(&pixels[0][0], 1, width());
            file.writePixels(height());

            console->debug("Writing EXR image took: {} seconds.", (timer.lap()/1000.f));
			return true;
        }
        catch (const exception &e)
        {
            console->error("ERROR: Unable to write image file \"{}\": {}", filename, e.what());
            return false;
        }
    }
    else
    {
        // convert floating-point image to 8-bit per channel with dithering
        vector<unsigned char> data(size()*3, 0);

        Timer timer;
        // convert 3-channel pfm data to 4-channel internal representation
        parallel_for(0, height(), [this,img,&data,dither](int y)
        {
            for (int x = 0; x < width(); ++x)
            {
                Color4 c = (*img)(x, y);
                if (dither)
                {
                    int xmod = x % 256;
                    int ymod = y % 256;
                    float ditherValue = (dither_matrix256[xmod + ymod * 256] / 65536.0f - 0.5f) / 255.0f;
                    c += Color4(Color3(ditherValue), 0.0f);
                }

                // convert to [0-255] range
                c = (c * 255.0f).max(0.0f).min(255.0f);

                data[3 * x + 3 * y * width() + 0] = (unsigned char) c[0];
                data[3 * x + 3 * y * width() + 1] = (unsigned char) c[1];
                data[3 * x + 3 * y * width() + 2] = (unsigned char) c[2];
            }
        });
        console->debug("Tonemapping to 8bit took: {} seconds.", (timer.elapsed()/1000.f));

        if (extension == "ppm")
            return writePPMImage(filename.c_str(), width(), height(), 3, &data[0]);
        else if (extension == "png")
            return stbi_write_png(filename.c_str(), width(), height(),
                                  3, &data[0], sizeof(unsigned char)*width()*3) != 0;
        else if (extension == "bmp")
            return stbi_write_bmp(filename.c_str(), width(), height(), 3, &data[0]) != 0;
        else if (extension == "tga")
            return stbi_write_tga(filename.c_str(), width(), height(), 3, &data[0]) != 0;
        else if (extension == "jpg" || extension == "jpeg")
            return stbi_write_jpg(filename.c_str(), width(), height(), 3, &data[0], 100) != 0;
        else
            throw invalid_argument("Could not determine desired file type from extension.");
    }
}


// local functions
namespace
{


// Taken from http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
const Matrix3f XYZD65TosRGB(
	(Matrix3f() << 3.2406f, -1.5372f, -0.4986f,
		-0.9689f,  1.8758f,  0.0415f,
		0.0557f, -0.2040f,  1.0570f).finished());

const Matrix3f XYZD50ToXYZD65(
	(Matrix3f() << 0.9555766f, -0.0230393f, 0.0631636f,
		-0.0282895f,  1.0099416f, 0.0210077f,
		0.0122982f, -0.0204830f, 1.3299098f).finished());

// Taken from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
const Matrix3f XYZD50TosRGB(
	(Matrix3f() << 3.2404542f, -1.5371385f, -0.4985314f,
		-0.9692660f,  1.8760108f,  0.0415560f,
		0.0556434f, -0.2040259f,  1.0572252).finished());

Matrix3f computeCameraToXYZD50(const tinydng::DNGImage &param)
{
	//
	// The full DNG color-correction model is described in the
	// "Mapping Camera Color Space to CIE XYZ Space" section of the DNG spec.
	//
	// Let n be the dimensionality of the camera color space (usually 3 or 4).
	// Let CM be the n-by-3 matrix interpolated from the ColorMatrix1 and ColorMatrix2 tags.
	// Let CC be the n-by-n matrix interpolated from the CameraCalibration1 and CameraCalibration2 tags (or identity matrices, if the signatures don't match).
	// Let AB be the n-by-n matrix, which is zero except for the diagonal entries, which are defined by the AnalogBalance tag.
	// Let RM be the 3-by-n matrix interpolated from the ReductionMatrix1 and ReductionMatrix2 tags.
	// Let FM be the 3-by-n matrix interpolated from the ForwardMatrix1 and ForwardMatrix2 tags.

	// TODO: the color correction code below is not quite correct

	// if the ForwardMatrix is included:
	if (false)//param.has_forward_matrix2)
	{
		auto FM((Matrix3f() << param.forward_matrix2[0][0], param.forward_matrix2[0][1], param.forward_matrix2[0][2],
			                   param.forward_matrix2[1][0], param.forward_matrix2[1][1], param.forward_matrix2[1][2],
							   param.forward_matrix2[2][0], param.forward_matrix2[2][1], param.forward_matrix2[2][2]).finished());
		auto CC((Matrix3f() << param.camera_calibration2[0][0], param.camera_calibration2[0][1], param.camera_calibration2[0][2],
							   param.camera_calibration2[1][0], param.camera_calibration2[1][1], param.camera_calibration2[1][2],
							   param.camera_calibration2[2][0], param.camera_calibration2[2][1], param.camera_calibration2[2][2]).finished());
		auto AB = Vector3f(param.analog_balance[0], param.analog_balance[1], param.analog_balance[2]).asDiagonal();

		Vector3f CameraNeutral(param.as_shot_neutral[0],
		                       param.as_shot_neutral[1],
		                       param.as_shot_neutral[2]);
		Vector3f ReferenceNeutral = (AB * CC).inverse() * CameraNeutral;
		auto D = (ReferenceNeutral.asDiagonal()).inverse();
		auto CameraToXYZ = FM * D * (AB * CC).inverse();

		return CameraToXYZ;
	}
	else
	{
		auto CM((Matrix3f() << param.color_matrix2[0][0], param.color_matrix2[0][1], param.color_matrix2[0][2],
			                   param.color_matrix2[1][0], param.color_matrix2[1][1], param.color_matrix2[1][2],
				               param.color_matrix2[2][0], param.color_matrix2[2][1], param.color_matrix2[2][2]).finished());

		auto CameraToXYZ = CM.inverse();

		return CameraToXYZ;

	}
}


HDRImage develop(vector<float> & raw,
                 const tinydng::DNGImage & param1,
                 const tinydng::DNGImage & param2)
{
	Timer timer;

	int width = param1.width;
	int height = param1.height;
	int blackLevel = param1.black_level[0];
	int whiteLevel = param1.white_level[0];
	Vector2i redOffset(param1.active_area[1] % 2, param1.active_area[0] % 2);

	HDRImage developed(width, height);

	Matrix3f CameraToXYZD50 = computeCameraToXYZD50(param2);
	Matrix3f CameraTosRGB = XYZD50TosRGB * CameraToXYZD50;

	// Chapter 5 of DNG spec
	// Map raw values to linear reference values (i.e. adjust for black and white level)
	//
	// we also apply white balance before demosaicing here because it increases the
	// correlation between the color channels and reduces artifacts
	Vector3f wb(param2.as_shot_neutral[0], param2.as_shot_neutral[1], param2.as_shot_neutral[2]);
	const float invScale = 1.0f / (whiteLevel - blackLevel);
	parallel_for(0, developed.height(), [&developed,&raw,blackLevel,invScale,&wb](int y)
	{
		for (int x = 0; x < developed.width(); x++)
		{
			float v = clamp((raw[y * developed.width() + x] - blackLevel)*invScale, 0.f, 1.f);
			Vector3f rgb = Vector3f(v,v,v);
			rgb = rgb.cwiseQuotient(wb);
			developed(x,y) = Color4(rgb(0),rgb(1),rgb(2),1.f);
		}
	});

	// demosaic
//	developed.demosaicLinear(redOffset);
//	developed.demosaicGreenGuidedLinear(redOffset);
//	developed.demosaicMalvar(redOffset);
	developed.demosaicAHD(redOffset, XYZD50ToXYZD65 * CameraToXYZD50);

	// color correction
	// also undo the white balance since the color correction matrix already includes it
	parallel_for(0, developed.height(), [&developed,&CameraTosRGB,&wb](int y)
	{
		for (int x = 0; x < developed.width(); x++)
		{
			Vector3f rgb(developed(x,y).r, developed(x,y).g, developed(x,y).b);
			rgb = rgb.cwiseProduct(wb);
			Vector3f sRGB = CameraTosRGB * rgb;
			developed(x,y) = Color4(sRGB.x(),sRGB.y(),sRGB.z(),1.f);
		}
	});

	spdlog::get("console")->debug("Developing DNG image took {} seconds.", (timer.elapsed()/1000.f));
	return developed;
}


inline unsigned short endianSwap(unsigned short val)
{
	unsigned short ret;

	unsigned char *buf = reinterpret_cast<unsigned char *>(&ret);

	unsigned short x = val;
	buf[1] = static_cast<unsigned char>(x);
	buf[0] = static_cast<unsigned char>(x >> 8);

	return ret;
}


// The decode functions below are adapted from syoyo's dng2exr, in the tinydng library within the
// ext subfolder

//
// Decode 12bit integer image into floating point HDR image
//
void decode12BitToFloat(vector<float> &image, unsigned char *data, int width, int height, bool swapEndian)
{
	Timer timer;

	int offsets[2][2] = {{0, 1}, {1, 2}};
	int bitShifts[2] = {4, 0};

	image.resize(static_cast<size_t>(width * height));

	parallel_for(0, height, [&image,width,&offsets,&bitShifts,data,swapEndian](int y)
	{
		for (int x = 0; x < width; x++)
		{
			unsigned char buf[3];

			// Calculate load address for 12bit pixel(three 8 bit pixels)
			int n = int(y * width + x);

			// 24 = 12bit * 2 pixel, 8bit * 3 pixel
			int n2 = n % 2;           // used for offset & bitshifts
			int addr3 = (n / 2) * 3;  // 8bit pixel pos
			int odd = (addr3 % 2);

			int bit_shift;
			bit_shift = bitShifts[n2];

			int offset[2];
			offset[0] = offsets[n2][0];
			offset[1] = offsets[n2][1];

			if (swapEndian)
			{
				// load with short byte swap
				if (odd)
				{
					buf[0] = data[addr3 - 1];
					buf[1] = data[addr3 + 2];
					buf[2] = data[addr3 + 1];
				}
				else
				{
					buf[0] = data[addr3 + 1];
					buf[1] = data[addr3 + 0];
					buf[2] = data[addr3 + 3];
				}
			}
			else
			{
				buf[0] = data[addr3 + 0];
				buf[1] = data[addr3 + 1];
				buf[2] = data[addr3 + 2];
			}
			unsigned int b0 = static_cast<unsigned int>(buf[offset[0]] & 0xff);
			unsigned int b1 = static_cast<unsigned int>(buf[offset[1]] & 0xff);

			unsigned int val = (b0 << 8) | b1;
			val = 0xfff & (val >> bit_shift);

			image[static_cast<size_t>(y * width + x)] = static_cast<float>(val);
		}
	});

	spdlog::get("console")->debug("decode12BitToFloat took: {} seconds.", (timer.lap() / 1000.f));
}

//
// Decode 14bit integer image into floating point HDR image
//
void decode14BitToFloat(vector<float> &image, unsigned char *data, int width, int height, bool swapEndian)
{
	Timer timer;

	int offsets[4][3] = {{0, 0, 1}, {1, 2, 3}, {3, 4, 5}, {5, 5, 6}};
	int bitShifts[4] = {2, 4, 6, 0};

	image.resize(static_cast<size_t>(width * height));

	parallel_for(0, height, [&image,width,&offsets,&bitShifts,data,swapEndian](int y)
	{
		for (int x = 0; x < width; x++)
		{
			unsigned char buf[7];

			// Calculate load address for 14bit pixel(three 8 bit pixels)
			int n = int(y * width + x);

			// 56 = 14bit * 4 pixel, 8bit * 7 pixel
			int n4 = n % 4;           // used for offset & bitshifts
			int addr7 = (n / 4) * 7;  // 8bit pixel pos
			int odd = (addr7 % 2);

			int offset[3];
			offset[0] = offsets[n4][0];
			offset[1] = offsets[n4][1];
			offset[2] = offsets[n4][2];

			int bit_shift;
			bit_shift = bitShifts[n4];

			if (swapEndian)
			{
				// load with short byte swap
				if (odd)
				{
					buf[0] = data[addr7 - 1];
					buf[1] = data[addr7 + 2];
					buf[2] = data[addr7 + 1];
					buf[3] = data[addr7 + 4];
					buf[4] = data[addr7 + 3];
					buf[5] = data[addr7 + 6];
					buf[6] = data[addr7 + 5];
				}
				else
				{
					buf[0] = data[addr7 + 1];
					buf[1] = data[addr7 + 0];
					buf[2] = data[addr7 + 3];
					buf[3] = data[addr7 + 2];
					buf[4] = data[addr7 + 5];
					buf[5] = data[addr7 + 4];
					buf[6] = data[addr7 + 7];
				}
			}
			else
			{
				memcpy(buf, &data[addr7], 7);
			}
			unsigned int b0 = static_cast<unsigned int>(buf[offset[0]] & 0xff);
			unsigned int b1 = static_cast<unsigned int>(buf[offset[1]] & 0xff);
			unsigned int b2 = static_cast<unsigned int>(buf[offset[2]] & 0xff);

			// unsigned int val = (b0 << 16) | (b1 << 8) | b2;
			// unsigned int val = (b2 << 16) | (b0 << 8) | b0;
			unsigned int val = (b0 << 16) | (b1 << 8) | b2;
			// unsigned int val = b2;
			val = 0x3fff & (val >> bit_shift);

			image[static_cast<size_t>(y * width + x)] = static_cast<float>(val);
		}
	});

	spdlog::get("console")->debug("decode14BitToFloat took: {} seconds.", (timer.lap() / 1000.f));
}

//
// Decode 16bit integer image into floating point HDR image
//
void decode16BitToFloat(vector<float> &image, unsigned char *data, int width, int height, bool swapEndian)
{
	Timer timer;

	image.resize(static_cast<size_t>(width * height));
	unsigned short *ptr = reinterpret_cast<unsigned short *>(data);

	parallel_for(0, height, [&image,width,ptr,swapEndian](int y)
	{
		for (int x = 0; x < width; x++)
		{
			unsigned short val = ptr[y * width + x];
			if (swapEndian)
				val = endianSwap(val);

			// range will be [0, 65535]
			image[static_cast<size_t>(y * width + x)] = static_cast<float>(val);
		}
	});

	spdlog::get("console")->debug("decode16BitToFloat took: {} seconds.", (timer.lap() / 1000.f));
}

char get_colorname(int c)
{
	switch (c)
	{
		case 0:
			return 'R';
		case 1:
			return 'G';
		case 2:
			return 'B';
		case 3:
			return 'C';
		case 4:
			return 'M';
		case 5:
			return 'Y';
		case 6:
			return 'W';
		default:
			return '?';
	}
}

void printImageInfo(const tinydng::DNGImage & image)
{
	auto console = spdlog::get("console");
	console->debug("width = {}.", image.width);
	console->debug("width = {}.", image.width);
	console->debug("height = {}.", image.height);
	console->debug("bits per pixel = {}.", image.bits_per_sample);
	console->debug("bits per pixel(original) = {}", image.bits_per_sample_original);
	console->debug("samples per pixel = {}", image.samples_per_pixel);
	console->debug("sample format = {}", image.sample_format);

	console->debug("version = {}", image.version);

	for (int s = 0; s < image.samples_per_pixel; s++)
	{
		console->debug("white_level[{}] = {}", s, image.white_level[s]);
		console->debug("black_level[{}] = {}", s, image.black_level[s]);
	}

	console->debug("tile_width = {}", image.tile_width);
	console->debug("tile_length = {}", image.tile_length);
	console->debug("tile_offset = {}", image.tile_offset);
	console->debug("tile_offset = {}", image.tile_offset);

	console->debug("cfa_layout = {}", image.cfa_layout);
	console->debug("cfa_plane_color = {}{}{}{}",
	               get_colorname(image.cfa_plane_color[0]),
	               get_colorname(image.cfa_plane_color[1]),
	               get_colorname(image.cfa_plane_color[2]),
	               get_colorname(image.cfa_plane_color[3]));
	console->debug("cfa_pattern[2][2] = \n {}, {},\n {}, {}",
	               image.cfa_pattern[0][0],
	               image.cfa_pattern[0][1],
	               image.cfa_pattern[1][0],
	               image.cfa_pattern[1][1]);

	console->debug("active_area = \n {}, {},\n {}, {}",
	               image.active_area[0],
	               image.active_area[1],
	               image.active_area[2],
	               image.active_area[3]);

	console->debug("calibration_illuminant1 = {}", image.calibration_illuminant1);
	console->debug("calibration_illuminant2 = {}", image.calibration_illuminant2);

	console->debug("color_matrix1 = ");
	for (size_t k = 0; k < 3; k++)
		console->debug("{} {} {}",
		               image.color_matrix1[k][0],
		               image.color_matrix1[k][1],
		               image.color_matrix1[k][2]);

	console->debug("color_matrix2 = ");
	for (size_t k = 0; k < 3; k++)
		console->debug("{} {} {}",
		               image.color_matrix2[k][0],
		               image.color_matrix2[k][1],
		               image.color_matrix2[k][2]);

	if (true)//image.has_forward_matrix2)
	{
		console->debug("forward_matrix1 found = ");
		for (size_t k = 0; k < 3; k++)
			console->debug("{} {} {}",
			               image.forward_matrix1[k][0],
			               image.forward_matrix1[k][1],
			               image.forward_matrix1[k][2]);
	}
	else
		console->debug("forward_matrix2 not found!");

	if (true)//image.has_forward_matrix2)
	{
		console->debug("forward_matrix2 found = ");
		for (size_t k = 0; k < 3; k++)
			console->debug("{} {} {}",
			               image.forward_matrix2[k][0],
			               image.forward_matrix2[k][1],
			               image.forward_matrix2[k][2]);
	}
	else
		console->debug("forward_matrix2 not found!");

	console->debug("camera_calibration1 = ");
	for (size_t k = 0; k < 3; k++)
		console->debug("{} {} {}",
		               image.camera_calibration1[k][0],
		               image.camera_calibration1[k][1],
		               image.camera_calibration1[k][2]);

	console->debug("orientation = {}", image.orientation);

	console->debug("camera_calibration2 = ");
	for (size_t k = 0; k < 3; k++)
		console->debug("{} {} {}",
		               image.camera_calibration2[k][0],
		               image.camera_calibration2[k][1],
		               image.camera_calibration2[k][2]);

	if (image.has_analog_balance)
		console->debug("analog_balance = {} , {} , {}",
		               image.analog_balance[0],
		               image.analog_balance[1],
		               image.analog_balance[2]);
	else
		console->debug("analog_balance not found!");

	if (image.has_as_shot_neutral)
		console->debug("as_shot_neutral = {} , {} , {}",
		               image.as_shot_neutral[0],
		               image.as_shot_neutral[1],
		               image.as_shot_neutral[2]);
	else
		console->debug("shot_neutral not found!");
}

} // namespace