/*
 * =====================================================================================
 *
 *       Filename:  2jjalloc.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2021年03月05日 19时22分41秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "2jjalloc.h"
#include <vector>
#include <iostream>

using namespace std;


int main() {
    int ia[5] = {0,1, 2, 3, 4};
    unsigned int i;
    vector<int, JJ::allocator<int>> iv(ia, ia +5);
    for (i = 0; i < iv.size(); i++) {
        cout << iv[i] <<  ' ';
    }
    cout << endl;
    return 0;
}

