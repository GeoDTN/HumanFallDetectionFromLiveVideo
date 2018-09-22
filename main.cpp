#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <CStdLib>
#include <CTime>
#include <IOStream>
using namespace std;
int  main() {
    //srand((int)time(nullptr));
srand((int) time(nullptr));
int dice = (rand() % 6 ) + 1;
cout << "Dice: " << dice << endl;
}
