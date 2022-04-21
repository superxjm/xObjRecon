#include "stdafx.h"
#include "Parse.h"

Parse::Parse()
{
}

const Parse & Parse::get()
{
  static const Parse instance;
  return instance;
}

int Parse::arg(int argc, char** argv, const char* str, std::string &val) const
{
    int index = findArg(argc, argv, str) + 1;

    if(index > 0 && index < argc)
    {
        val = argv[index];
    }

    return index - 1;
}

int Parse::arg(int argc, char** argv, const char* str, float &val) const
{
    int index = findArg(argc, argv, str) + 1;

    if(index > 0 && index < argc)
    {
        val = atof(argv[index]);
    }

    return index - 1;
}

int Parse::arg(int argc, char** argv, const char* str, int &val) const
{
    int index = findArg(argc, argv, str) + 1;

    if(index > 0 && index < argc)
    {
        val = atoi(argv[index]);
    }

    return index - 1;
}

std::string Parse::shaderDir() const
{
    std::string currentVal = std::string("D:\\xjm\\xObjRecon\\xObjRecon\\xSurfelFusion\\Shaders");

    assert(pangolin::FileExists(currentVal) && "Shader directory not found!");

    return currentVal;
}

std::string Parse::baseDir() const
{
    char buf[256];
    int length = GetModuleFileNameA(NULL, buf, sizeof(buf));
    std::string currentVal;
    currentVal.append((char *)&buf, length);

    currentVal = currentVal.substr(0, currentVal.rfind("\\build\\"));
    return currentVal;
}

int Parse::findArg(int argc, char** argv, const char* argument_name) const
{
    for(int i = 1; i < argc; ++i)
    {
        // Search for the string
        if(strcmp(argv[i], argument_name) == 0)
        {
            return i;
        }
    }
    return -1;
}



