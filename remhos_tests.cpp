#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <unistd.h>

#include "general/communication.hpp"
using mfem::real_t;

int remhos(int, char *[], double &);

///////////////////////////////////////////////////////////////////////////////
template <class T>
std::enable_if_t<!std::numeric_limits<T>::is_integer, bool>
AlmostEq(T x, T y, T tolerance = 10.0*std::numeric_limits<T>::epsilon())
{
   const T neg = std::abs(x - y);
   constexpr T min = std::numeric_limits<T>::min();
   constexpr T eps = std::numeric_limits<T>::epsilon();
   const T min_abs = std::min(std::abs(x), std::abs(y));
   if (std::abs(min_abs) == 0.0) { return neg < eps; }
   return (neg / (1.0 + std::max(min, min_abs))) < tolerance;
}

///////////////////////////////////////////////////////////////////////////////
struct Test
{
   static constexpr const char *binary = "remhos";
   static constexpr const char *common =
      "-dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 -no-vis -vs 1";
   std::string name, mesh, options;
   const real_t result = std::numeric_limits<real_t>::signaling_NaN();
   Test(const char *name, const char *mesh, const char *extra, real_t result)
      : name(name), mesh(mesh), options(std::string(extra) + " " + common),
        result(result) {}
   std::string Command() const
   {
      return std::string(binary) + " " + mesh + " " + options;
   }
};

///////////////////////////////////////////////////////////////////////////////
const Test runs[] =
{
   {
      "quad-14-1-2", // #0
      "-m ./data/inline-quad.mesh", "-p 14 -rs 1 -o 2",
      0.0971135253228191
   },
   {
      "quad-14-4-3", // #1
      "-m ./data/inline-quad.mesh", "-p 14 -rs 4 -o 3",
      0.0930984398474033
   },
   {
      "quad-14-4-4", // #2
      "-m ./data/inline-quad.mesh", "-p 14 -rs 4 -o 4",
      0.0923763049829040
   },
   {
      "cube01-10-1-2", // #3
      "-m ./data/cube01_hex.mesh", "-p 10 -rs 1 -o 2",
      0.1197219672304615
   },
   {
      "quad-pa-14-1-2", // #4
      "-m ./data/inline-quad.mesh", "-pa -p 14 -rs 1 -o 2",
      0.0971135252706320
   },
   {
      "quad-pa-14-4-2", // #5
      "-m ./data/inline-quad.mesh", "-pa -p 14 -rs 4 -o 2",
      0.0918571775667415
   },
   {
      "cube01-pa-10-2-3", // #6
      "-m ./data/cube01_hex.mesh", "-pa -p 10 -rs 2 -o 3",
      0.1162578642211454
   },
   {
      "star-q2-pa-14-1-3", // #7
      "-m ./data/star-q2.mesh", "-pa -p 14 -rs 1 -o 3",
      0.7975688080682961
   },
   {
      "debug-quad-14-1-2", // #8
      "-m ./data/inline-quad.mesh", "-d debug -pa -p 14 -rs 1 -o 2",
      0.0971135252706320
   },
#ifdef MFEM_USE_CUDA
   {
      "cuda-quad-14-1-2",
      "-m ./data/inline-quad.mesh",
      "-d cuda -pa -p 14 -rs 1 -o 2",
      0.0971135252706320
   },
#endif
};
constexpr int N_TESTS = sizeof(runs) / sizeof(Test);

///////////////////////////////////////////////////////////////////////////////
using args_ptr_t = std::vector<std::unique_ptr<char[]>>;
using args_t = std::vector<char*>;

///////////////////////////////////////////////////////////////////////////////
int RemhosTest(const Test & test)
{
   static args_ptr_t args_ptr;
   args_t args;

   std::istringstream iss(test.Command());

   std::string token;
   while (iss >> token)
   {
      auto arg_ptr = std::make_unique<char[]>(token.size() + 1);
      std::memcpy(arg_ptr.get(), token.c_str(), token.size() + 1);
      arg_ptr[token.size()] = '\0';
      args.push_back(arg_ptr.get());
      args_ptr.emplace_back(std::move(arg_ptr));
   }
   args.push_back(nullptr);

   double final_mass_u{};
   remhos(args.size()-1, args.data(), final_mass_u);

   if (AlmostEq(final_mass_u, test.result)) { return EXIT_SUCCESS; }

   mfem::err << "❌ " << test.name
             << std::fixed << std::setprecision(16)
             << ": final_mass_u: " << final_mass_u
             << " vs. " << test.result
             << std::endl;
   return EXIT_FAILURE;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) try
{
   mfem::Mpi::Init(argc, argv);

   int opt;
   int test = -1;
   auto show_usage = [](const int ret = EXIT_FAILURE)
   {
      printf("Usage: program [-a <arg>] [-b <arg>] [-h]\n");
      printf("  -t <test>  Optional test number \n");
      printf("  -h         Show this help message\n");
      exit(ret);
   };

   while ((opt = getopt(argc, argv, "t:h")) != -1)
   {
      switch (opt)
      {
         case 't': test = std::atoi(optarg); break;
         case 'h': show_usage(EXIT_SUCCESS);
         default: show_usage(EXIT_FAILURE);
      }
   }

   if (test >= 0 && test < N_TESTS) { return RemhosTest(runs[test]); }

   for (auto & run : runs)
   {
      if (RemhosTest(run) != EXIT_SUCCESS) { return EXIT_FAILURE; }
   }
   return EXIT_SUCCESS;
}
catch (std::exception& e)
{
   std::cerr << "\033[31m..xxxXXX[ERROR]XXXxxx.." << std::endl;
   std::cerr << "\033[31m{}" << e.what() << std::endl;
   return EXIT_FAILURE;
}
