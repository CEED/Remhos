#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <unistd.h>

// #define DBG_COLOR ::debug::kCyan
// #include "debug.hpp"

int remhos(int, char *[], double &);

//// ///////////////////////////////////////////////////////////////////////////
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
   static constexpr const char *binary = "remhos ";
   static constexpr const char *common = "-no-vis -vs 1";
   std::string mesh, options;
   const double result = 0.0;
   Test(const char * msh, const char * extra, double result)
      : mesh(msh), options(std::string(extra) + common), result(result) {}
   std::string Command() const { return binary + mesh + options; }
};

///////////////////////////////////////////////////////////////////////////////
const Test runs[] =
{
   {
      // #0
      "-m ./data/inline-quad.mesh ",
      "-p 14 -rs 1 -o 2 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.09711395400387984
   },
   {
      // #1, 65536
      "-m ./data/inline-quad.mesh ",
      "-p 14 -rs 4 -o 3 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.0930984399257905
   },
   {
      // #2, 102400
      // RHS   kernel time: 5.4988707
      // L2inv kernel time: 0.94180588
      // LO    kernel time: 0.028575583
      // FCT   kernel time: 0.0095909583
      // Total kernel time: 5.5370372
      "-m ./data/inline-quad.mesh ",
      "-p 14 -rs 4 -o 4 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.09237630484178257
   },
   {
      // #3
      "-m ./data/cube01_hex.mesh ",
      "-p 10 -rs 1 -o 2 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.11972857593296446
   },
   {
      // #4 Partial assembly
      "-m ./data/inline-quad.mesh ",
      "-pa -p 14 -rs 1 -o 2 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.09711395400387984
   },
   {
      // #5 Partial assembly
      "-m ./data/inline-quad.mesh ",
      "-pa -p 14 -rs 4 -o 2 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.09185717760402806
   },
   {
      // #7: 3D PA
      "-m ./data/cube01_hex.mesh ",
      "-pa -p 10 -rs 3 -o 3 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 1 ",
      0.11601536511552431
   },
   {
      // #8
      "-m ./data/star-q2.mesh ",
      "-pa -p 14 -rs 1 -o 3 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.8069675186775516
   },
   {
      // #9, 🔥 debug device 🔥 (last or !dup)
      "-m ./data/inline-quad.mesh ",
      "-d debug -pa -p 14 -rs 1 -o 2 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.09711395400387984
   },
#ifdef MFEM_USE_CUDA
   {
      // #10
      "-m ./data/inline-quad.mesh ",
      "-d cuda -pa -p 14 -rs 1 -o 2 -dt -1.0 -tf 0.5 -ho 3 -lo 5 -fct 2 -ms 5 ",
      0.09711395400387984
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

   // dbg("final_mass_u: {} vs. {}", final_mass_u, test.result);

   if (AlmostEq(final_mass_u, test.result)) { return std::cout << "✅" << std::endl, EXIT_SUCCESS; }

   return  std::cout << "❌" << std::endl, EXIT_FAILURE;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) try
{
   // dbgClearScreen(), dbg();

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