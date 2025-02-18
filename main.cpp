#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>

#define STARTS_WITH 786

using namespace std;

namespace Array {
    /*1. Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
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
    

    /*2. You are given an array prices where prices[i] is the price of a given stock on the ith day.
    You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
    Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.*/
    namespace buySellStock {
        int maxProfit(vector<int> prices) {
            if(prices.size() == 0) 
                return 0;
            int maxProfit = 0, minPrice = prices[0];
            const int len = prices.size();
            for(int i = 1; i < len; ++i) {
                minPrice = min(minPrice, prices[i]);
            maxProfit = max(maxProfit,prices[i] - minPrice);    
            }
            return maxProfit;
        }
    }


    /*3. Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct. */
    namespace containsDuplicate {
        bool useSet(vector<int> nums) {
            return set(nums.begin(), nums.end()).size() != nums.size();
        }
        bool useUnorderedMap(vector<int> nums) {
            unordered_map<int, int> mp;
            for(const int& num: nums) {
                if(mp.count(num))
                    return true;
                ++mp[num];
            }
            return false;
        }
    }

    /*Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]. The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. You must write an algorithm that runs in O(n) time and without using the division operation.  */
    namespace productOfArrayExceptSelf {
        vector<int> productExceptSelf(vector<int> nums) {
            const int len = nums.size();

            vector<int> res(len, 1);
    
            int left = 1, right = 1;
    
            for(int i = 0; i < len; ++i) {
                res[i] *= left;
                res[len-1-i] *= right;
                left *= nums[i];
                right *= nums[len-1-i];
            }
    
            return res;
        }
    }

    /*Given an integer array nums, find the subarray with the largest sum, and return its sum.*/
    namespace maximumSubArray {
        int getMaximumSubArray(const vector<int>& arr) {
            if(arr.empty())
                return 0;
            int maxi = arr.front();
            int currMaxi = 0;

            for(const int& num: arr) {
                currMaxi = max(0, currMaxi);
                currMaxi += num;
                maxi = max(maxi, currMaxi);
            }

            return maxi;
        }
    }
 
}

int main( ){
    Array::twoSum::loop( { 2, 7, 11, 15 }, 9 );
    cout << Array::buySellStock::maxProfit( { 7,1,5,3,6,4 } );
    cout << endl << Array::containsDuplicate::useSet( { 1, 2, 3, 1 } );
    cout << endl << Array::containsDuplicate::useUnorderedMap( { 1, 2, 3, 1 } );
    
    cout << endl;
    for(const int& num: Array::productOfArrayExceptSelf::productExceptSelf({1, 2, 3, 4}))
        cout << num << " ";

    cout << "\nMax subArray: " << Array::maximumSubArray::getMaximumSubArray({4, -2, 3, 4});

    return 0;   
}