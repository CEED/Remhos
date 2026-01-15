#include <iostream>
#include <unistd.h>

int remhos(int argc, char *argv[], double &final_mass_u);

int main(int argc, char* argv[]) try
{
   double result;
   return remhos(argc, argv, result);
}
catch (std::exception& e)
{
   std::cerr << "\033[31m..xxxXXX[ERROR]XXXxxx.." << std::endl;
   std::cerr << "\033[31m{}" << e.what() << std::endl;
   return EXIT_FAILURE;
}