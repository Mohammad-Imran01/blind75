#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

#define STARTS_WITH int(786)

using namespace std;

namespace Array {
    /*Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    You can return the answer in any order.*/
    namespace twoSum {
        // Loop based two sum solution with O(n^2) time complexity and constant space complexity.
        vector<int> loop(vector<int> nums, int target) {
            vector<int> res{-1, -1};
            const int len = nums.size();
            for(int i = 0; i < len-1; ++i) {
                for(int j = i+1; j < len; ++j) {
                    if(nums[i] + nums[j] == target)
                        return {i, j};
                }
            }
            return res;
        }
        // Solve the two sum problem using hash map with O(n) time complexity and O(n) space complexity.
        pair<int, int> hashMap( vector<int> nums, int target ) {
            unordered_map<int, int> mp;
            const int len = nums.size();
            for(int i = 0; i < len; ++i) {
                if(mp.count(target - nums[i]))
                    return {mp[target-nums[i]], i};
                mp[nums[i]] = i;
            }
            return {-1, -1};    
        }
    }
}

int main( ){
    Array::twoSum::loop( { 2, 7, 11, 15 }, 9 );
    return 0;   
}