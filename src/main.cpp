#include <iostream>
#include <stdio.h>

int main(void) {
    int x[3][4] = {{0,1,2,3}, {4,5,6,7}, {8,9,10,11}};
    std::cout << x[8] << std::endl;
    return 0;
}