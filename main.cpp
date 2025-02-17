#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

namespace Array {
    namespace twoSum {
        void loop( vector<int> nums, int target ) {
            const int len = nums.size( );
            for(int i = 0; i < len; ++i){
                for(int j = i + 1; j < len; ++j){
                    if( nums[ i ] + nums[ j ] == target ){
                        cout << i << " " << j << endl;
                        return;
                    }
                }
            }
        }
    }
}

int main( ){
    Array::twoSum::loop( { 2, 7, 11, 15 }, 9 );
    return 0;   
}