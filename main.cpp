#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <cinttypes>

#define STARTS_WITH 786

using namespace std;
using VI2 = vector<vector<int>>;

//------------------------======================================================
// Contains several functions to solve problems of the Array section from the Blind 75 problem set.
namespace Array {

    /* 1. Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
       You may assume that each input would have exactly one solution, and you may not use the same element twice.
       You can return the answer in any order. */

    // Time Complexity: O(n^2), Space Complexity: O(1)
    vector<int> twoSumLoop(vector<int> nums, int target) {
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

    // Time Complexity: O(n), Space Complexity: O(n)
    pair<int, int> twoSumHashMap(vector<int> nums, int target) {
        unordered_map<int, int> mp;
        const int len = nums.size();
        for(int i = 0; i < len; ++i) {
            if(mp.count(target - nums[i]))
                return {mp[target-nums[i]], i};
            mp[nums[i]] = i;
        }
        return {-1, -1};    
    }
    

    /* 2. You are given an array prices where prices[i] is the price of a given stock on the ith day.
       You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
       Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0. */

    // Time Complexity: O(n), Space Complexity: O(1)
    int maxProfit(vector<int> prices) {
        if(prices.size() == 0) 
            return 0;
        int maxProfit = 0, minPrice = prices[0];
        const int len = prices.size();
        for(int i = 1; i < len; ++i) {
            minPrice = min(minPrice, prices[i]);
            maxProfit = max(maxProfit, prices[i] - minPrice);    
        }
        return maxProfit;
    }


    /* 3. Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct. */

    // Time Complexity: O(n log n), Space Complexity: O(n)
    bool containsDuplicateUseSet(vector<int> nums) {
        return set<int>(nums.begin(), nums.end()).size() != nums.size();
    }

    // Time Complexity: O(n), Space Complexity: O(n)
    bool containsDuplicateUseUnorderedMap(vector<int> nums) {
        unordered_map<int, int> mp;
        for(const int& num: nums) {
            if(mp.count(num))
                return true;
            ++mp[num];
        }
        return false;
    }

    /* 4. Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
       The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. You must write an algorithm that runs in O(n) time and without using the division operation. */

    // Time Complexity: O(n), Space Complexity: O(1) (excluding the output array)
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

    /* 5. Given an integer array nums, find the subarray with the largest sum, and return its sum. */

    // Time Complexity: O(n), Space Complexity: O(1)
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
 
    /* 6. Suppose an array of length n sorted in ascending order is rotated between 1 and n times. Return the smallest element in O(log n) time. */

    // Time Complexity: O(log n), Space Complexity: O(1)
    void findMinInRotatedSortedArray(const vector<int>& arr) {
        int mini = INT_MAX;
        int left = 0, right = arr.size()-1;

        while (left <= right) {
            int mid = left + (right-left)/2;
            if(arr.at(left) <= arr.at(right)) {
                mini = min(mini, arr.at(left));
                break;
            }
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

    /* 7. Given a list of numbers, return a list of triplets such that the sum of the triplet is zero. */

    // Time Complexity: O(n^2 log n), Space Complexity: O(n)
    vector<vector<int>> threeSumUseSet(vector<int> nums) {
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

    // Time Complexity: O(n^2), Space Complexity: O(1) (excluding the output array)
    vector<vector<int>> threeSumLoopOnly(vector<int> nums) {
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


    /* 8. Also Known as the trapping the rainwater problem. Given an array of heights, return the maximum possible area. */

    // Time Complexity: O(n^2), Space Complexity: O(1)
    int maxAreaTwoLoops(vector<int> heights) {
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

    // Time Complexity: O(n), Space Complexity: O(1)
    int maxAreaSingleTraversal(vector<int> heights) {
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
} // End of namespace Array
//------------------------======================================================

//------------------------======================================================
// Contains several functions to solve problems of the Binary section from the Blind 75 problem set.
namespace Binary {

    /* 1. Add two integers without using the addition operator.
       Approach: Use bitwise operations to simulate addition.
       - Calculate the carry using AND operation.
       - Use XOR to add the numbers without considering the carry.
       - Shift the carry left by 1 and repeat until there is no carry.
       Time Complexity: O(log n), Space Complexity: O(1) */
    int addWithoutAdditionOperator(int a, int b) {
        while(b) {
            int c = a & b; // Carry: both bits are 1, so take it and move further
            a ^= b;        // Sum of bits without considering carry
            b = c << 1;     // Shift carry to the left by 1
        }
        return a;
    }

    /* 2. Count the number of set bits (1s) in the binary representation of a number.
       Approach: Iterate through each bit of the number and count the set bits.
       Time Complexity: O(log n), Space Complexity: O(1) */
    int countSetBits(int n) {
        int res = 0;
        while(n) {
            res += (n & 1); // Check if the least significant bit is set
            n >>= 1;         // Right shift to check the next bit
        }
        return res;
    }

    /* 3. Generate a vector where each element represents the count of set bits for numbers from 0 to n.
       Approach: Use the `countSetBits` function to compute the number of set bits for each number.
       Time Complexity: O(n log n), Space Complexity: O(n) */
    vector<int> generateZeroToNSetBits(int n) {
        ++n; // Include n in the result
        vector<int> res(n, 0);

        for(int i = 0; i < n; ++i) 
            res[i] = countSetBits(i); // Compute set bits for each number

        return res;
    }

    /* 4. Find the missing number in a sequence of numbers from 0 to n.
       Approach: Use a temporary array to mark the presence of numbers and find the missing one.
       Time Complexity: O(n), Space Complexity: O(n) */
    int missingNumberLoop(vector<int> nums) {
        vector<int> temp(nums.size() + 1, -1); // Temporary array to mark presence
        for(const int& num: nums)
            temp[num] = num; // Mark the number as present
        for(int i = 0; i < temp.size(); ++i)
            if(temp.at(i) == -1) // Find the missing number
                return i;
        return -1; // If no missing number found
    }

    /* 5. Find the missing number in a sequence of numbers from 0 to n using XOR.
       Approach: Use XOR to cancel out all numbers present in the array, leaving the missing number.
       Time Complexity: O(n), Space Complexity: O(1) */
    int missingNumXor(vector<int> nums) {
        int res = nums.size(); // Initialize result with n

        for(int i = 0; i < nums.size(); ++i)
            res = i ^ res ^ nums[i]; // XOR all indices and numbers

        return res; // The missing number
    }

    /* 6. Reverse the bits of a 32-bit unsigned integer.
       Approach: Iterate through each bit of the number and construct the reversed number.
       Time Complexity: O(1) (since it's always 32 bits), Space Complexity: O(1) */
    uint32_t reverseBits(uint32_t n) {
        uint32_t res = 0;
        const int len = sizeof(n) * 8; // Number of bits in the integer

        for(int i = 0; i < len; ++i) {
            res = (res << 1) | (n & 1); // Shift result left and add the least significant bit of n
            n >>= 1; // Right shift n to process the next bit
        }

        return res;
    }
} // namespace Binary
//------------------------======================================================


//------------------------======================================================
namespace dp {
    //Time Complexity: O(2^n)
    //Space Complexity: 𝑂(𝑛)
    int climbStairs(int n) {
        if(n <= 1) return 0;
        return climbStairs(n-1) + climbStairs(n-2);
    }
    int climbStairMemoized(int n) {
        vector<int> memo(n+1, 0);
        memo[0] = memo[1] = 1;
        for(int i = 2; i <= n; ++i) 
            memo[i] = memo[i-1] + memo[i-2];
        return memo.back();
    }

    // Number of minimum coins needed to sum up a target, using any given coin any number of times.
    int coinChangeHelp(vector<int>& coins, vector<vector<int>> &dp, int amount, int ind) {
        if(amount == 0) return 0;
        if(amount < 0) return 1e8;
        if(ind < 0 || ind >= coins.size()) return 1e8;

        if(dp[ind][amount] != -1)
            return dp[ind][amount];
        

        int take = 1e8;

        if(amount >= coins[ind])
            take = 1 + min(coinChangeHelp(coins, dp, amount - coins[ind], ind), coinChangeHelp(coins, dp, amount - coins[ind], ind+1));

        return dp[ind][amount] = min(take, coinChangeHelp(coins, dp, amount, ind+1));
    }
    int coinChange(vector<int>& coins, int amount) {
        vector<vector<int>> dp(int(coins.size()+1), vector<int>(amount+1, -1));
        const int res = coinChangeHelp(coins, dp, amount, 0);
        return res >= 1e8 ? -1:res;
    }




} // namespace dp

//------------------------======================================================






int main( ){
    
    

    return 0;   
}