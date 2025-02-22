#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <unordered_set>

#define STARTS_WITH 786

using namespace std;

// Contains several funtions to solve problems of Array section from the Blind 75 problem set.
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
 
    /*Suppose an array of length n sorted in ascending order is rotated between 1 and n times. Return the smallest element in O(log n) time*/
    namespace minInRotatedSortedArray {
        void bSearch(const vector<int>& arr) {
            int mini = INT_MAX;
            int left = 0, right = arr.size()-1;

            while (left <= right) {
                int mid = left + (right-left)/2;
                if(arr.at(left) <= arr.at(right)) {
                    mini = min(mini, arr.at(left));
                    break;
                } // left < right: left is ans
                if(arr.at(left) <= arr.at(mid)) {
                    mini = min(mini, arr.at(left));
                    left = mid+1;
                } else {
                    mini = min(mini, arr.at(mid));
                    right = mid-1;
                }
            }

            cout << "\nMin in rotated sorted array: " << mini;
        }
    }

    /*4. Given a list of numbers, return a list of triplets such that the sum of the triplet is zero.*/
    namespace threeSum {
        // Uses loop to traverse and a set to store unique triplets. Time complexity O(n^2 log n), extra space O(n).
        vector<vector<int>> useSet(vector<int> nums) {
            set<vector<int>> resSet;
            sort(begin(nums), end(nums));
            const int len = nums.size();
            for(int i = 0; i < len-2; ++i) {
                int left = i+1, right = len-1;
                while(left < right) {
                    const int curr = nums[i] + nums[left] + nums[right];
                    if(curr > 0) --right;
                    else if (curr < 0) ++left;
                    else {
                        resSet.insert({nums[i], nums[left++], nums[right--]});
                    }
                }
            }
            return vector<vector<int>>(resSet.begin(), resSet.end());
        } 
        // Uses loop to traverse. Time complexity O(n^2) and constant space.
        vector<vector<int>> loopOnly(vector<int> nums) {
            if(nums.size() < 3)
                return {};
            sort(nums.begin(), nums.end());
            const int len = nums.size();
            vector<vector<int>> res;
            for(int i = 0; i < len-2; ++i) {
                while((i > 0) && (i < len) && (nums[i] == nums[i-1])) ++i;
                int left = i+1, right = len-1;
                while(left < right) {
                    const int curr = nums[i] + nums[left] + nums[right];
                    if(curr > 0) --right;
                    else if(curr < 0) ++left;
                    else {
                        res.push_back({nums[i], nums[left], nums[right]});

                        while((left < right) && (nums[left] == nums[left+1])) ++left;
                        while((right > left) && (nums[right] == nums[right-1])) --right;

                        ++left;
                        --right;
                    }
                }
            }

            return res;
        }
    }


    // Also Known as trapping the rainwater problem. Given an array of heights return the maximum possible area
    namespace maxArea {
        // traverse through the array and returns the calc max area Time: O(n^2) space is constant.
        int twoLoops(vector<int> heights) {
            if(heights.size() < 2)
                return 0;

            const int len = heights.size();
            
            int maxi = INT_MIN;

            for(int i = 0; i < len-1; ++i) {
                for(int j = i+1; j < len; ++j) {
                    const int curr = min(heights[i], heights[j]) * (j-i);
                    maxi = max(maxi, curr);
                }
            }

            return maxi;
        }

        // finds solution in single traversal Time: O(n) and space is constant.
        int singleTraversal(vector<int> heights) {
            int left = 0, right = heights.size()-1;
            int maxi = INT_MIN;

            while(left < right) {
                const int l = heights[left];
                const int r = heights[right];

                if(l < r) {
                    maxi = max(maxi, l * (right-left));
                    ++left;
                } else {
                    maxi = max(maxi, r * (right-left));
                    --right;
                }
            }

            return maxi;
        }
    }
} /* 
namespace for core array problems*/

// Contains several funtions to solve problems of Binary section from the Blind 75 problem set.
namespace Binary
{
    // uses a simple while loop. Time O(log n), Space constant.
    int addWithoutAdditionOperator(int a, int b) {
        while(b) {
            int c = a&b; //carry: both is 1's so take it and move further
            a ^= b;
            b = c << 1;
        }
        return a;
    }
    int countSetBits(int n) {
        int res = 0;
        while(n) {
            res += (n&1);
            n >>= 1;
        }
        return res;
    }

    vector<int> generateZeroToNSetBits(int n) {
        ++n;
        vector<int> res(n, 0);

        for(int i = 0; i < n; ++i) 
            res[i] = countSetBits(i);

        return res;
    }
} // namespace Binary

int main( ){
    // Array::twoSum::loop( { 2, 7, 11, 15 }, 9 );
    // cout << Array::buySellStock::maxProfit( { 7,1,5,3,6,4 } );
    // cout << endl << Array::containsDuplicate::useSet( { 1, 2, 3, 1 } );
    // cout << endl << Array::containsDuplicate::useUnorderedMap( { 1, 2, 3, 1 } );
    
    // cout << endl;
    // for(const int& num: Array::productOfArrayExceptSelf::productExceptSelf({1, 2, 3, 4}))
    //     cout << num << " ";

    // cout << "\nMax subArray: " << Array::maximumSubArray::getMaximumSubArray({4, -2, 3, 4});

    // Array::minInRotatedSortedArray::bSearch({11, 22, 33, 44, 10});

    // Array::threeSum::loopOnly({1,2,3,4,-8,2,-4,-5});

    // cout << "\nSet bits: " << Binary::countSetBits(5) << "\n";

    // auto res = Binary::generateZeroToNSetBits(5);


    // for(const auto& num: res) {
    //     cout << num << ", ";
    // }


    

    return 0;   
}