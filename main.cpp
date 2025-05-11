#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <cinttypes>
#include <string.h>
#include <functional>
#include <stack>
#include <queue>
#include <utility>

#define STARTS_WITH 786

using namespace std;

// VI2 is a type alias for a two-dimensional std::vector of integers, where both the inner and outer vectors use the default std::allocator. It simplifies the declaration of nested vectors for managing 2D integer data in C++.
using VI1 = vector<int>;
using VI2 = vector<VI1>;
using VC1 = vector<char>;
using VC2 = vector<VC1>;

//------------------------======================================================
// Contains several functions to solve problems of the Array section from the Blind 75 problem set.
namespace Array
{

    /* 1. Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
       You may assume that each input would have exactly one solution, and you may not use the same element twice.
       You can return the answer in any order. */

    // Time Complexity: O(n^2), Space Complexity: O(1)
    vector<int> twoSumLoop(vector<int> nums, int target)
    {
        vector<int> res{-1, -1};
        const int len = nums.size();
        for (int i = 0; i < len - 1; ++i)
        {
            for (int j = i + 1; j < len; ++j)
            {
                if (nums[i] + nums[j] == target)
                    return {i, j};
            }
        }
        return res;
    }

    // Time Complexity: O(n), Space Complexity: O(n)
    pair<int, int> twoSumHashMap(vector<int> nums, int target)
    {
        unordered_map<int, int> mp;
        const int len = nums.size();
        for (int i = 0; i < len; ++i)
        {
            if (mp.count(target - nums[i]))
                return {mp[target - nums[i]], i};
            mp[nums[i]] = i;
        }
        return {-1, -1};
    }

    /* 2. You are given an array prices where prices[i] is the price of a given stock on the ith day.
       You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
       Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0. */

    // Time Complexity: O(n), Space Complexity: O(1)
    int maxProfit(vector<int> prices)
    {
        if (prices.size() == 0)
            return 0;
        int maxProfit = 0, minPrice = prices[0];
        const int len = prices.size();
        for (int i = 1; i < len; ++i)
        {
            minPrice = min(minPrice, prices[i]);
            maxProfit = max(maxProfit, prices[i] - minPrice);
        }
        return maxProfit;
    }

    /* 3. Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct. */

    // Time Complexity: O(n log n), Space Complexity: O(n)
    bool containsDuplicateUseSet(vector<int> nums)
    {
        return set<int>(nums.begin(), nums.end()).size() != nums.size();
    }

    // Time Complexity: O(n), Space Complexity: O(n)
    bool containsDuplicateUseUnorderedMap(vector<int> nums)
    {
        unordered_map<int, int> mp;
        for (const int &num : nums)
        {
            if (mp.count(num))
                return true;
            ++mp[num];
        }
        return false;
    }

    /* 4. Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
       The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. You must write an algorithm that runs in O(n) time and without using the division operation. */

    // Time Complexity: O(n), Space Complexity: O(1) (excluding the output array)
    vector<int> productExceptSelf(vector<int> nums)
    {
        const int len = nums.size();
        vector<int> res(len, 1);

        int left = 1, right = 1;

        for (int i = 0; i < len; ++i)
        {
            res[i] *= left;
            res[len - 1 - i] *= right;
            left *= nums[i];
            right *= nums[len - 1 - i];
        }

        return res;
    }

    /* 5. Given an integer array nums, find the subarray with the largest sum, and return its sum. */

    // Time Complexity: O(n), Space Complexity: O(1)
    int getMaximumSubArray(const vector<int> &arr)
    {
        if (arr.empty())
            return 0;
        int maxi = arr.front();
        int currMaxi = 0;

        for (const int &num : arr)
        {
            currMaxi = max(0, currMaxi);
            currMaxi += num;
            maxi = max(maxi, currMaxi);
        }

        return maxi;
    }

    /* 6. Suppose an array of length n sorted in ascending order is rotated between 1 and n times. Return the smallest element in O(log n) time. */

    // Time Complexity: O(log n), Space Complexity: O(1)
    void findMinInRotatedSortedArray(const vector<int> &arr)
    {
        int mini = INT_MAX;
        int left = 0, right = arr.size() - 1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            if (arr.at(left) <= arr.at(right))
            {
                mini = min(mini, arr.at(left));
                break;
            }
            if (arr.at(left) <= arr.at(mid))
            {
                mini = min(mini, arr.at(left));
                left = mid + 1;
            }
            else
            {
                mini = min(mini, arr.at(mid));
                right = mid - 1;
            }
        }

        cout << "\nMin in rotated sorted array: " << mini;
    }

    /* 7. Given a list of numbers, return a list of triplets such that the sum of the triplet is zero. */

    // Time Complexity: O(n^2 log n), Space Complexity: O(n)
    vector<vector<int>> threeSumUseSet(vector<int> nums)
    {
        set<vector<int>> resSet;
        sort(begin(nums), end(nums));
        const int len = nums.size();
        for (int i = 0; i < len - 2; ++i)
        {
            int left = i + 1, right = len - 1;
            while (left < right)
            {
                const int curr = nums[i] + nums[left] + nums[right];
                if (curr > 0)
                    --right;
                else if (curr < 0)
                    ++left;
                else
                {
                    resSet.insert({nums[i], nums[left++], nums[right--]});
                }
            }
        }
        return vector<vector<int>>(resSet.begin(), resSet.end());
    }

    // Time Complexity: O(n^2), Space Complexity: O(1) (excluding the output array)
    vector<vector<int>> threeSumLoopOnly(vector<int> nums)
    {
        if (nums.size() < 3)
            return {};
        sort(nums.begin(), nums.end());
        const int len = nums.size();
        vector<vector<int>> res;
        for (int i = 0; i < len - 2; ++i)
        {
            while ((i > 0) && (i < len) && (nums[i] == nums[i - 1]))
                ++i;
            int left = i + 1, right = len - 1;
            while (left < right)
            {
                const int curr = nums[i] + nums[left] + nums[right];
                if (curr > 0)
                    --right;
                else if (curr < 0)
                    ++left;
                else
                {
                    res.push_back({nums[i], nums[left], nums[right]});

                    while ((left < right) && (nums[left] == nums[left + 1]))
                        ++left;
                    while ((right > left) && (nums[right] == nums[right - 1]))
                        --right;

                    ++left;
                    --right;
                }
            }
        }

        return res;
    }

    /* 8. Also Known as the trapping the rainwater problem. Given an array of heights, return the maximum possible area. */

    // Time Complexity: O(n^2), Space Complexity: O(1)
    int maxAreaTwoLoops(vector<int> heights)
    {
        if (heights.size() < 2)
            return 0;

        const int len = heights.size();
        int maxi = INT_MIN;

        for (int i = 0; i < len - 1; ++i)
        {
            for (int j = i + 1; j < len; ++j)
            {
                const int curr = min(heights[i], heights[j]) * (j - i);
                maxi = max(maxi, curr);
            }
        }

        return maxi;
    }

    // Time Complexity: O(n), Space Complexity: O(1)
    int maxAreaSingleTraversal(vector<int> heights)
    {
        int left = 0, right = heights.size() - 1;
        int maxi = INT_MIN;

        while (left < right)
        {
            const int l = heights[left];
            const int r = heights[right];

            if (l < r)
            {
                maxi = max(maxi, l * (right - left));
                ++left;
            }
            else
            {
                maxi = max(maxi, r * (right - left));
                --right;
            }
        }

        return maxi;
    }
} // End of namespace Array
//------------------------======================================================

//------------------------======================================================
// Contains several functions to solve problems of the Binary section from the Blind 75 problem set.
namespace Binary
{

    /* 1. Add two integers without using the addition operator.
       Approach: Use bitwise operations to simulate addition.
       - Calculate the carry using AND operation.
       - Use XOR to add the numbers without considering the carry.
       - Shift the carry left by 1 and repeat until there is no carry.
       Time Complexity: O(log n), Space Complexity: O(1) */
    int addWithoutAdditionOperator(int a, int b)
    {
        while (b)
        {
            int c = a & b; // Carry: both bits are 1, so take it and move further
            a ^= b;        // Sum of bits without considering carry
            b = c << 1;    // Shift carry to the left by 1
        }
        return a;
    }

    /* 2. Count the number of set bits (1s) in the binary representation of a number.
       Approach: Iterate through each bit of the number and count the set bits.
       Time Complexity: O(log n), Space Complexity: O(1) */
    int countSetBits(int n)
    {
        int res = 0;
        while (n)
        {
            res += (n & 1); // Check if the least significant bit is set
            n >>= 1;        // Right shift to check the next bit
        }
        return res;
    }

    /* 3. Generate a vector where each element represents the count of set bits for numbers from 0 to n.
       Approach: Use the `countSetBits` function to compute the number of set bits for each number.
       Time Complexity: O(n log n), Space Complexity: O(n) */
    vector<int> generateZeroToNSetBits(int n)
    {
        ++n; // Include n in the result
        vector<int> res(n, 0);

        for (int i = 0; i < n; ++i)
            res[i] = countSetBits(i); // Compute set bits for each number

        return res;
    }

    /* 4. Find the missing number in a sequence of numbers from 0 to n.
       Approach: Use a temporary array to mark the presence of numbers and find the missing one.
       Time Complexity: O(n), Space Complexity: O(n) */
    int missingNumberLoop(vector<int> nums)
    {
        vector<int> temp(nums.size() + 1, -1); // Temporary array to mark presence
        for (const int &num : nums)
            temp[num] = num; // Mark the number as present
        for (int i = 0; i < temp.size(); ++i)
            if (temp.at(i) == -1) // Find the missing number
                return i;
        return -1; // If no missing number found
    }

    /* 5. Find the missing number in a sequence of numbers from 0 to n using XOR.
       Approach: Use XOR to cancel out all numbers present in the array, leaving the missing number.
       Time Complexity: O(n), Space Complexity: O(1) */
    int missingNumXor(vector<int> nums)
    {
        int res = nums.size(); // Initialize result with n

        for (int i = 0; i < nums.size(); ++i)
            res = i ^ res ^ nums[i]; // XOR all indices and numbers

        return res; // The missing number
    }

    /* 6. Reverse the bits of a 32-bit unsigned integer.
       Approach: Iterate through each bit of the number and construct the reversed number.
       Time Complexity: O(1) (since it's always 32 bits), Space Complexity: O(1) */
    uint32_t reverseBits(uint32_t n)
    {
        uint32_t res = 0;
        const int len = sizeof(n) * 8; // Number of bits in the integer

        for (int i = 0; i < len; ++i)
        {
            res = (res << 1) | (n & 1); // Shift result left and add the least significant bit of n
            n >>= 1;                    // Right shift n to process the next bit
        }

        return res;
    }
} // namespace Binary
//------------------------======================================================

//------------------------======================================================
namespace dp
{
    // Time Complexity: O(2^n)
    // Space Complexity: 𝑂(𝑛)
    int climbStairs(int n)
    {
        if (n <= 1)
            return 0;
        return climbStairs(n - 1) + climbStairs(n - 2);
    }
    int climbStairMemoized(int n)
    {
        vector<int> memo(n + 1, 0);
        memo[0] = memo[1] = 1;
        for (int i = 2; i <= n; ++i)
            memo[i] = memo[i - 1] + memo[i - 2];
        return memo.back();
    }

    // Number of minimum coins needed to sum up a target, using any given coin any number of times.
    int coinChangeHelp(vector<int> &coins, vector<vector<int>> &dp, int amount, int ind)
    {
        if (amount == 0)
            return 0;
        if (amount < 0)
            return 1e8;
        if (ind < 0 || ind >= coins.size())
            return 1e8;

        if (dp[ind][amount] != -1)
            return dp[ind][amount];

        int take = 1e8;

        if (amount >= coins[ind])
            take = 1 + min(coinChangeHelp(coins, dp, amount - coins[ind], ind), coinChangeHelp(coins, dp, amount - coins[ind], ind + 1));

        return dp[ind][amount] = min(take, coinChangeHelp(coins, dp, amount, ind + 1));
    }
    int coinChange(vector<int> &coins, int amount)
    {
        vector<vector<int>> dp(int(coins.size() + 1), vector<int>(amount + 1, -1));
        const int res = coinChangeHelp(coins, dp, amount, 0);
        return res >= 1e8 ? -1 : res;
    }

    int lengthOfLIS(VI1 &nums, VI2 &dp, int ind, int lastInd)
    {
        if (ind >= nums.size() || ind < 0)
            return 0;
        if (dp[ind][lastInd + 1] != -1)
            return dp[ind][lastInd + 1];
        int take = 0, ignore = lengthOfLIS(nums, dp, ind + 1, lastInd);
        if (lastInd == -1 || nums[ind] > nums[lastInd])
            take = 1 + lengthOfLIS(nums, dp, ind + 1, ind);
        return dp[ind][lastInd + 1] = max(take, ignore);
    }
    // Given an array return the length of the strictly increasing subsequence.
    int lengthOfLIS(vector<int> nums)
    {
        if (nums.size() < 2)
            return nums.size();
        const int len = nums.size();
        VI2 dp(len, VI1(len + 1, -1));

        return lengthOfLIS(nums, dp, 0, -1);
    }

    int lengthOfLISLoop(vector<int> nums)
    {
        if (nums.size() < 2)
            return nums.size();
        const int len = nums.size();
        vector<int> dp(len, 1);

        for (int curr = 1; curr < len; ++curr)
        {
            for (int last = 0; last < curr; ++last)
            {
                if (nums[curr] > nums[last] && dp[curr] <= dp[last])
                    dp[curr] = 1 + dp[last];
            }
        }

        return *max_element(dp.begin(), dp.end());
    }

    // longest common substring
    int LCS(string s1, string s2, int i1 = 0, int i2 = 0)
    {
        if (i1 >= s1.size() || i2 >= s2.size() || i1 < 0 || i2 < 0)
            return 0;
        if (s1[i1] == s2[i2])
            return 1 + LCS(s1, s2, i1 + 1, i2 + 1);
        return max(LCS(s1, s2, i1 + 1, i2), LCS(s1, s2, i1, i2 + 1));
    }
    int LCS(string s1, string s2)
    {
        const int len1 = s1.size();
        const int len2 = s2.size();
        if (s1.empty() || s2.empty())
            return 0;
        VI2 dp(len1 + 1, VI1(len2 + 1, 0));
        for (int i = 1; i <= len1; ++i)
        {
            for (int j = 1; j <= len2; ++j)
            {
                if (s1[i - 1] == s2[j - 1])
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                else
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }

        return dp.back().back();
    }

    bool wordBreak(string s, vector<string> &wordDict)
    {
        const int len = s.size();
        vector<bool> dp(len + 1, false);
        dp.back() = true;

        for (int i = len - 1; i >= 0; --i)
        {
            for (const string &word : wordDict)
            {
                const int n = word.size();
                if (i + n <= len && s.substr(i, n) == word)
                    dp[i] = dp[i + n];
                if (dp[i])
                    break; // Stop once dp[i] is true
            }
        }

        return dp.front();
    }
    //-----------------------------------------------------------------------------

    void combinationSumHelp(VI2 &res, VI1 &curr, const VI1 &candidates, int ind, int target)
    {
        if (target == 0)
        {
            res.push_back(curr);
            return;
        }
        if (ind >= candidates.size() || target < candidates[ind])
            return;

        curr.push_back(candidates[ind]);
        combinationSumHelp(res, curr, candidates, ind, target - candidates[ind]);
        curr.pop_back();
        combinationSumHelp(res, curr, candidates, ind + 1, target);
    }

    // Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.
    // The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.
    // The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
    VI2 combinationSum(const VI1 &nums, int target)
    {
        VI2 res;
        VI1 curr;
        combinationSumHelp(res, curr, nums, 0, target);

        return res;
    }

    //     You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

    // Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

    int houseRobber(const VI1 &nums, VI1 &dp, int ind = 0)
    {
        if (ind < 0 || ind >= nums.size())
            return 0;

        if (dp[ind] != -1)
            return dp[ind];

        const int res = max(houseRobber(nums, dp, ind + 1),
                            nums[ind] + houseRobber(nums, dp, ind + 2));

        return dp[ind] = res;
    }

    // house robber circular
    int rob(vector<int> &nums)
    {
        if (nums.empty())
            return 0;
        if (nums.size() < 2)
            return nums.front();

        VI1 dp(nums.size(), -1);
        const int res1 = houseRobber(nums, dp, 1);

        dp = VI1(nums.size(), -1);
        nums.pop_back();
        const int res2 = houseRobber(nums, dp, 0);

        return max(res1, res2);
    }

    // ------------------------------============================================================--------
    // You have intercepted a secret message encoded as a string of numbers. The message is decoded via the following mapping:
    // "AAJF" with the grouping (1, 1, 10, 6)
    // "KJF" with the grouping (11, 10, 6)
    // The grouping (1, 11, 06) is invalid because "06" is not a valid code (only "6" is valid).
    class DecodeWays
    {
    public:
        int dp[101];
        int numDecodings(string s)
        {
            memset(dp, -1, sizeof(dp));
            return solve(s, 0);
        }
        bool vald(const string &s)
        {
            if (s.empty() || s.front() == '0')
                return false;
            const int val = stoi(s);

            return val >= 1 && val <= 26;
        }
        int solve(const string &s, int ind)
        {
            if (ind == s.size())
                return 1;
            if (s[ind] == '0')
                return 0;

            if (dp[ind] != -1)
                return dp[ind];

            int one = solve(s, ind + 1);
            int two = 0;
            if (ind < s.size() - 1 && vald(s.substr(ind, 2)))
            {
                two = solve(s, ind + 2);
            }

            return dp[ind] = one + two;
        }
    };

    //--------------------------------------------===============================------------
    class UniquePaths
    {
    private:
        int dp[101][101];

        int Solve(int m, int n, int i, int j)
        {
            if (i < 0 || j < 0)
                return 0;
            if (i == 0 && j == 0)
                return 1;
            if (dp[i][j] != -1)
                return dp[i][j];

            return Solve(m, n, i - 1, j) + Solve(m, n, i, j - 1);
        }

    public:
        UniquePaths()
        {
            memset(dp, -1, sizeof(dp));
        }
        int Solve(int m, int n)
        {
            return Solve(m, n, m - 1, n - 1);
        }
    };

    class CanJump
    {
    public:
        bool canJump(vector<int> &nums, int i = 0)
        {
            const int len = nums.size();

            vector<int> dp(len, 0);
            dp[0] = 1;

            for (int i = 0; i < len - 1; ++i)
            {
                for (int j = 1; j <= nums[i] && i + j < len; ++j)
                    dp[i + j] = dp[i + j - 1];
            }
            return dp.back();
        }
    };
} // namespace dp

//------------------------======================================================
namespace Graph
{
    struct Node
    {
        int val;
        vector<Node *> neighbors;
    };
    class CloneGraph
    {
        unordered_map<Node *, Node *> mp;

    public:
        Node *solution(Node *node)
        {
            if (!node)
                return nullptr;
            if (mp.count(node))
                return mp[node];
            Node *copy = new Node;
            copy->val = node->val;
            mp[node] = copy;

            for (Node *n : node->neighbors)
                copy->neighbors.push_back(solution(n));
            return copy;
        }
    };

    /* There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. */
    class CourseSchedulex
    {
        VI1 graph[2001];
        int vis[2001];

    public:
        bool canTake(int courseCount, const VI2 &pre)
        {
            memset(vis, 0, sizeof(vis));

            for (const auto &p : pre)
                graph[p.front()].push_back(p.back());

            // to check if there is any cycle from a specific node
            function<bool(int)> dfs = [&](int node)
            {
                if (vis[node] == 1)
                    return true; // cycle found, you are visiting this node only;
                if (vis[node] == 2)
                    return false; // this node checked as not cyclic

                vis[node] = 1;

                for (const int &u : graph[node])
                {
                    if (dfs(u))
                        return true;
                }

                vis[node] = 2;

                return false;
            };

            for (int i = 0; i < courseCount; ++i)
            {
                if (dfs(i))
                    return false; // there is a cycle
            }
            return true; // can take all courses.
        }
    };

    // Provides a solution to determine all grid cells in a matrix from which water can flow to both the Pacific and Atlantic oceans.It uses depth - first search(DFS) to explore valid paths based on elevation constraints and returns the coordinates of such cells.class PacificAtlanticWaterFlow
    class PacificAtlanticWaterFlow
    {
    private:
        bool vis[201][201];
        int m, n;

        VI2 moves{{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

        bool isValid(int r, int c)
        {
            return r >= 0 && c >= 0 && r < m && c < n;
        }
        void help(const VI2 &graph, int r, int c, bool &pac, bool &atl)
        {
            if (!isValid(r, c) || vis[r][c])
                return;
            vis[r][c] = true;

            if (r == 0 || c == 0)
                pac = true;
            if (r == m - 1 || c == n - 1)
                atl = true;

            for (const auto &move : moves)
            {
                int row = r + move[0], col = c + move[1];
                if (isValid(row, col) && graph[row][col] <= graph[r][c])
                    help(graph, row, col, pac, atl);
            }
        }

    public:
        // Takes a 2D vector graph as input and returns a 2D vector of coordinates where water can flow to both the Pacific and Atlantic oceans. It iterates through each cell in the grid, using a helper function to determine if the cell satisfies the conditions for both oceans, and collects the valid coordinates in the result.
        VI2 solution(const VI2 &graph)
        {
            m = graph.size(), n = graph.front().size();
            VI2 res;
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    memset(vis, false, sizeof(vis));
                    bool pac = false, atl = false;
                    help(graph, i, j, pac, atl);
                    if (pac && atl)
                        res.push_back({i, j});
                }
            }
            return res;
        }
    };

    // ---------------------------------------------------------=====================================
    // The Graph::NumberOfIslands class provides functionality to calculate the number of connected components(islands) in a 2D grid represented as a binary matrix.It uses depth - first search(DFS) to traverse and mark visited cells, with the main logic implemented in the solution method.
    class NumberOfIslands
    {

        int m,
            n;
        VI2 moves{{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

    private:
        void help(VI2 &graph, int row, int col)
        {
            if (row >= m || col >= n || row < 0 || col < 0 || !graph[row][col])
                return;
            graph[row][col] = false;
            for (const auto &move : moves)
                help(graph, row + move[0], col + move[1]);
        }

    public:
        int solution(VI2 graph)
        {
            m = graph.size(), n = graph[0].size();
            int res{};
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    res += graph[i][j];
                    help(graph, i, j);
                }
            }

            return res;
        }
    };

    // --------------------------------------------------=========================================
    // Graph::LongestConsecutive is a class that provides functionality to solve the problem of finding the length of the longest consecutive sequence in a given list of integers.It includes a public method solve that takes a vector of integers, removes duplicates, sorts the sequence, and calculates the length of the longest consecutive subsequence.

    class LongestConsecutive
    {
    public:
        int
        solve(VI1 seq)
        {
            if (seq.empty())
                return 0;
            unordered_set<int> st(seq.begin(), seq.end());
            seq = VI1(st.begin(), st.end());
            sort(seq.begin(), seq.end());

            const int len = seq.size();

            int res = 1, curr = 1;

            for (int i = 1; i < len; ++i)
            {
                curr = (seq[i] - 1 == seq[i - 1]) ? curr + 1 : 1;
                res = max(res, curr);
            }

            return res;
        }
    };

    // The Graph::AlienDictionary class provides functionality to determine the order of characters in an alien language based on a given dictionary of words.It constructs a directed graph to represent character precedence, performs a depth - first search to detect cycles, and derives a valid character order if possible.class AlienDictionary
    class AlienDictionary
    {
    private:
        unordered_map<char, vector<char>> graph;
        int vis[26];
        string res;

    private:
        bool dfs(char node)
        {
            if (vis[node - 'a'] == 1)
                return true; // found loop
            if (vis[node - 'a'] == 2)
                return false; // ok done
            vis[node - 'a'] = 1;

            for (char c : graph[node])
                if (dfs(c))
                    return true;

            vis[node - 'a'] = 2;
            res += node;

            return false;
        }

    public:
        string findOrder(string dict[], int N, int K)
        {
            for (int i = 0; i < N; ++i)
                for (char c : dict[i])
                    graph[c] = vector<char>();

            for (int i = 0; i < N - 1; ++i)
            {
                string s1 = dict[i], s2 = dict[i + 1];
                const int len = min(s1.size(), s2.size());
                bool found = false;
                for (int j = 0; j < len; ++j)
                {
                    if (s1[j] != s2[j])
                    {
                        graph[s1[j]].push_back(s2[j]);
                        found = true;
                        break;
                    }
                }
                if (!found && s1.size() > s2.size())
                    return "";
            }
            for (auto it = graph.begin(); it != graph.end(); ++it)
                if (dfs(it->first))
                    return "";

            reverse(begin(res), end(res));

            return res;
        }
    };

    // Validates whether a given graph with n nodes and a list of edges forms a valid tree. It uses depth-first search (DFS) to check for cycles and ensures all nodes are connected, leveraging an adjacency list and a visited array for its operations.
    class ValidGraphTree
    {
    private:
        unordered_map<int, VI1> graph;
        int vis[101];

    private:
        bool dfs(int node, int par)
        {
            if (vis[node] == 1)
                return true;
            vis[node] = 1;

            for (const auto &__node : graph[node])
            {
                if (__node == par)
                    continue;
                dfs(__node, node);
            }
            return false;
        }

    public:
        bool validate(int n, VI2 edges)
        {
            if (n == 0)
                return false;
            memset(vis, 0, sizeof(vis));
            for (const VI1 &edge : edges)
            {
                graph[edge[0]].push_back(edge[1]);
                graph[edge[1]].push_back(edge[0]);
            }

            if (dfs(0, -1))
                return false;

            for (int i = 0; i < n; ++i)
            {
                if (vis[i] == 0)
                    return false;
            }

            return true;
        }
    };
    // calculate the number of connected components in an undirected graph.It uses depth - first search(DFS) to traverse the graph, represented as an adjacency list, and determines the number of components based on unvisited nodes.private : vector<bool> vis;
    class NumberOfComponent
    {
        unordered_map<int, vector<int>> graph;
        vector<bool> vis;
        int res = 0;

        void dfs(int node, int par)
        {
            if (vis[node])
                return;
            vis[node] = true;

            for (const auto &__node : graph[node])
            {
                if (__node == par)
                    continue;
                dfs(__node, node);
            }
        }

    public:
        int findNumberOfComponent(int E, int V, vector<vector<int>> &edges)
        {
            for (const auto &edge : edges)
            {
                graph[edge[0]].push_back(edge[1]);
                graph[edge[1]].push_back(edge[0]);
            }
            vis = vector<bool>(V, false);

            for (int i = 0; i < V; ++i)
            {
                if (!vis[i])
                {
                    ++res;
                    dfs(i, -1);
                }
            }
            return res;
        }
    };

} // graph

namespace overlapping
{
    // Merge a new interval into a list of existing intervals, ensuring that overlapping intervals are combined into a single continuous range.The solve method takes a vector of intervals and a new interval, processes them to merge overlaps, and returns the updated list of intervals.class InsertInterval
    class InsertInterval
    {
    public:
        VI2 solve(VI2 intervals, VI1 newInterval)
        {
            const int len = intervals.size();

            int ind = 0;
            VI2 res;

            while (ind < len && intervals[ind][1] < newInterval.front())
                res.push_back(intervals[ind++]);

            while (ind < len && intervals[ind][0] <= newInterval[1])
            {
                newInterval[0] = min(newInterval[0], intervals[ind][0]);
                newInterval[1] = max(newInterval[1], intervals[ind][1]);

                ++ind;
            }

            res.push_back(newInterval);

            while (ind < len)
                res.push_back(intervals[ind++]);

            return res;
        }
    };

    /*Merge overlapping intervals from a given list of intervals. Its solve method takes a 2D vector of intervals, merges overlapping ones, and returns the resulting list of merged intervals.*/
    class MergeInterval
    {
    public:
        VI2 solve(const VI2 &arr)
        {
            if (arr.empty())
                return VI2();
            VI2 res;
            const int len = arr.size();

            res.push_back(arr.front());

            for (int i = 1; i < len; ++i)
            {
                auto &last = res.back();
                if (arr[i][0] <= last[1])
                    last[1] = max(last[1], arr[i][1]);
                else
                    res.push_back(arr.at(i));
            }
            return res;
        }
    };

    // return the minimum number of intervals to remove from a collection of intervals (represented as VI2) to ensure no overlapping intervals remain. The method sorts the intervals by their end points and uses a greedy algorithm to determine the count of non-overlapping intervals.
    class EraseNonOverLapping
    {
    public:
        int solve(VI2 arr) noexcept
        {
            if (arr.empty())
                return 0;
            std::sort(begin(arr), end(arr), [](const VI1 &a, const VI1 &b)
                      { return a.back() < b.back(); });
            int count = 1;
            int lastEnd = arr.front().back();
            const int len = arr.size();
            for (int i = 1; i < len; ++i)
            {
                if (arr[i][0] >= lastEnd)
                {
                    lastEnd = arr[i][1];
                    ++count;
                }
            }
            return len - count;
        }
    };

    // class provides methods to solve the problem of finding the missing and repeating numbers in an array of integers. It includes two implemented approaches: solveNestedLoop, which uses nested loops to count occurrences, and solveExtraSpace, which uses an auxiliary array for tracking counts, with a partially implemented solveSum method commented out.
    class MissingAndRepeatingNumber
    {
    public:
        VI1 solveNestedLoop(const VI1 &arr)
        {
            if (arr.empty())
                return {};
            int rep = 0, mis = 0;
            for (int i = 1; i <= arr.size(); ++i)
            {
                int cnt = 0;
                for (int j = 0; j < arr.size(); ++j)
                {
                    if (i == arr[j])
                        ++cnt;
                }
                if (cnt == 0)
                    mis = i;
                if (cnt == 2)
                    rep = i;
            }
            return {mis, rep};
        }
        VI1 solveExtraSpace(const VI1 &arr)
        {
            if (arr.empty())
                return {};
            int rep = 0, mis = 0;
            VI1 temp(arr.size() + 1, 0);
            for (const int &a : arr)
                ++temp[a];
            for (int i = 1; i <= arr.size(); ++i)
            {
                if (temp[i] > 1)
                    rep = i;
                if (temp[i] < 1)
                    mis = i;
            }
            return {mis, rep};
        }
        // VI1 solveSum(const VI1& arr) {
        //     if(arr.empty())
        // }
    };
};
namespace LinkedList
{
    // Represents a node in a linked list, containing a pointer to the next node(next) and an integer value(val).
    struct Node
    {
    public:
        Node *next;
        int val;

    public:
        Node(Node *__next) : val(0), next(__next) {}
        Node(int __val = 0, Node *__next = nullptr) : val(__val), next(__next) {}
    };

    // Provides multiple methods to reverse a linked list
    class Reverse
    {
    public:
        Node *reverseWithVector(Node *head)
        {
            if (!head)
                return head;
            vector<Node *> temp;
            while (head)
            {
                temp.emplace_back(head);
                head = head->next;
            }

            reverse(temp.begin(), temp.end());

            for (int i = 1; i < temp.size(); ++i)
            {
                temp[i - 1]->next = temp[i];
            }
            temp.back()->next = nullptr;
            return temp.front();
        }
        Node *reverseWithStack(Node *head)
        {
            if (!head)
                return head;
            stack<Node *> st;
            while (head)
            {
                st.push(head);
                head = head->next;
            }
            head = st.top();
            Node *temp = head;
            st.pop();
            while (st.size())
            {
                head->next = st.top();
                st.pop();
                head = head->next;
            }
            head->next = nullptr;

            return temp;
        }
        Node *reverseLoop(Node *head)
        {
            if (!head)
                return head;
            Node *prev = nullptr;
            while (head)
            {
                Node *next = head->next;
                head->next = prev;
                prev = head;
                head = next;
            }
            return prev;
        }
    };

    // Detect cycles in a linked list.It includes a solve method that uses the Floyd's Cycle Detection Algorithm (tortoise and hare approach) to determine if a cycle exists in the list by checking if two pointers meet.
    class DetectCycle
    {
    public:
        bool solve(Node *head)
        {
            auto *temp = head;
            while (temp && temp->next)
            {
                temp = temp->next->next;
                head = head->next;
                if (head == temp)
                    return true;
            }
            return false;
        }
    };

    // Merge two sorted linked lists into a single sorted linked list. It offers two methods: solveUsingVector, which uses a vector and sorting, and solveUsingTwoPointers, which employs a two-pointer technique for efficient merging.
    class MergeSortedLists
    {
        vector<Node *> nodes;
        void push(Node *head)
        {
            while (head)
            {
                nodes.push_back(head);
                head = head->next;
            }
        }

    public:
        Node *solveUsingVector(Node *h1, Node *h2)
        {
            if (!h1)
                return h2;
            if (!h2)
                return h1;

            push(h1);
            push(h2);

            std::sort(nodes.begin(), nodes.end(), [](Node *a, Node *b)
                      { return a->val < b->val; });
            for (int i = 1; i < nodes.size(); ++i)
                nodes[i - 1]->next = nodes[i];
            nodes.back()->next = nullptr;
            return nodes.front();
        }
        Node *solveUsingTwoPointers(Node *h1, Node *h2)
        {
            Node *root = new Node;
            Node *temp = root;

            while (h1 && h2)
            {
                if (h1->val <= h2->val)
                {
                    temp->next = h1;
                    h1 = h1->next;
                }
                else
                {
                    temp->next = h2;
                    h2 = h2->next;
                }
                temp = temp->next;
            }
            temp->next = h1 ? h1 : h2;

            return root->next;
        }

        // merges multiple sorted linked lists into a single sorted linked list using a two - pointer approach, returning the head of the resulting merged list.
        Node *mergeList(vector<Node *> nodes)
        {
            Node *res = nullptr;
            for (Node *&node : nodes)
                res = solveUsingTwoPointers(res, node);
            return res;
        }

        // Merges multiple sorted linked lists into a single sorted linked list. It uses a priority queue with a custom comparator to efficiently combine the nodes from the input lists and returns the head of the merged list.
        Node *mergeListWithPQueue(vector<Node *> nodes)
        {
            function<bool(Node *, Node *)> cmp = [](Node *a, Node *b)
            {
                return a->val > b->val;
            };
            priority_queue<Node *, vector<Node *>, decltype(cmp)> pq;

            for (auto node : nodes)
            {
                while (node)
                {
                    pq.push(node);
                    node = node->next;
                }
            }

            Node *root = new Node();
            auto *temp = root;

            while (pq.size())
            {
                temp->next = pq.top();
                temp = temp->next;
                pq.pop();
            }
            temp->next = nullptr;
            return root->next;
        }
    };

    Node *removeKthNodeFromLast(Node *head, int k)
    {
        head = new Node(head);
        Node
            *fast = head->next,
            *slow = head;
        while (fast)
        {
            fast = fast->next;
            if (k < 1)
                slow = slow->next;
            --k;
        }
        slow->next = slow->next->next;
        return head->next;
    }

    class ReorderList
    {
    public:
        void solve(Node *head)
        {
            if (!head || !head->next)
                return;

            auto *temp = head;
            stack<Node *> st;

            int len = 0;

            while (temp)
            {
                st.push(temp);
                temp = temp->next;
                ++len;
            }

            temp = head;
            for (int i = 0; i < len / 2; ++i)
            {
                auto *t = st.top();
                st.pop();
                t->next = temp->next;
                temp->next = t;
                temp = t->next;
            }
            if (temp)
                temp->next = nullptr;
        }
    };
}
// linkedlist

namespace Matrix
{
    // Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
    class TurnZero
    {
    public:
        void solve(VI2 mat)
        {
            if (mat.empty())
                return;

            const int
                m = mat.size(),
                n = mat.front().size();
            set<int> rows, cols;

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (mat[i][j] == 0)
                    {
                        rows.insert(i);
                        cols.insert(j);
                    }
                }
            }
            for (auto row : rows)
            {
                for (int col = 0; col < n; ++col)
                {
                    mat[row][col] = 0;
                }
            }
            for (auto col : cols)
            {
                for (int row = 0; row < m; ++row)
                {
                    mat[row][col] = 0;
                }
            }
        }
    };

    // Prints its elements in a spiral order if printResult is set to true.It processes the matrix by iterating through its boundaries in a clockwise manner, storing the elements in a 1D vector(VI1) before optionally printing them to the console.
    class PrintSpiral
    {
    public:
        void SpiralPrint(VI2 mat, bool printResult = true)
        {
            if (mat.empty())
                return;
            const int m = mat.size();
            const int n = mat.front().size();
            int
                top = 0,
                right = n - 1,
                bottom = m - 1,
                left = 0;
            const int len = m * n;

            VI1 res(len);
            int ind = 0;

            while (ind < len)
            {
                for (int col = left; col <= right && (ind < len); ++col)
                    res[ind++] = mat[top][col];
                ++top;

                for (int row = top; row <= bottom && (ind < len); ++row)
                    res[ind++] = mat[row][right];
                --right;

                for (int col = right; col >= left && (ind < len); --col)
                    res[ind++] = mat[bottom][col];
                --bottom;

                for (int row = bottom; row >= top && (ind < len); --row)
                    res[ind++] = mat[row][left];
                ++left;
            }

            if (printResult)
                for (int element : res)
                    std::cout << element << " ";
        }
    };

    // Rotate a square matrix 90 degrees clockwise.Its solve method takes a 2D vector(VI2) as input, performs the rotation, and updates the matrix in place.
    class Rotate90
    {
    public:
        void solve(VI2 mat)
        {
            if (mat.empty())
                return;

            const int n = mat.size();
            auto res = VI2(n, VI1(n, 0));

            int col = n - 1;
            for (int i = 0; i < n; ++i)
            {
                int r = 0;
                for (int j = 0; j < n; ++j)
                {
                    res[r++][col] = mat[i][j];
                }
                --col;
            }
            mat = res;
        }
    };

    // search for a given word in a 2D grid of characters. It uses a depth-first search (DFS) algorithm to determine if the word can be constructed by sequentially adjacent cells, ensuring no cell is reused in the same word path.

    class WordSearch
    {
    private:
        int m, n;

    private:
        bool dfs(VC2 board, string name, int r, int c, int ind)
        {
            if (ind >= name.size())
                return true;
            if (r < 0 || c < 0 || r >= m || c >= n || board[r][c] != name[ind])
                return false;

            board[r][c] = '@';

            bool found =
                dfs(board, name, r + 1, c, ind + 1) ||
                dfs(board, name, r - 1, c, ind + 1) ||
                dfs(board, name, r, c + 1, ind + 1) ||
                dfs(board, name, r, c - 1, ind + 1);

            board[r][c] = name[ind];

            return found;
        }

    public:
        bool find(VC2 board, string name)
        {
            if (board.empty() || name.empty())
                return false;
            m = board.size();
            n = board[0].size();

            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    if (board[i][j] == name.front() && dfs(board, name, i, j, 0))
                        return true;
            return false;
        }
    };
} // matrix

namespace String
{
    // calculates the length of the longest substring without repeating characters in a given string.It uses a sliding window technique with a boolean vector to track visited characters efficiently.
    class LongestNonRepeatingSubstring
    {
    public:
        int solve(string s)
        {
            vector<bool> vis(128, false);
            int res = 0;
            int left = 0;
            for (int right = 0; right < s.length(); ++right)
            {
                while (vis[s[right]])
                    vis[s[left++]] = false;

                vis[s[right]] = true;
                res = max(res, right + 1 - left);
            }
            return res;
        }
    };

    // determines the length of the longest substring of a given string s that can be transformed into a substring with all identical characters by replacing at most k characters.The implementation uses a sliding window technique to efficiently calculate the result
    class CharacterReplacement
    {
    public:
        int solve(string s, int k)
        {
            int res = 0, maxCount = 0, left = 0;
            int len = s.size();
            int arr[26];
            memset(arr, 0, sizeof(arr));

            for (int right = 0; right < len; ++right)
            {
                ++arr[s[right] - 'A'];
                maxCount = max(maxCount, arr[s[right] - 'A']);
                while ((right - left + 1) - maxCount > k)
                    --arr[s[left++] - 'A'];
                res = max(res, right - left + 1);
            }
            return res;
        }
    };

    class Anagram
    {
    public:
        bool solve(string s, string t)
        {
            if (s.size() != t.size())
                return false;
            int arr[26];
            memset(arr, 0, sizeof(arr));
            for (int i = 0; i < s.size(); ++i)
            {
                ++arr[s[i] - 'a'];
                --arr[t[i] - 'a'];
            }
            for (int i = 0; i < 26; ++i)
                if (arr[i])
                    return false;
            return true;
        }
    };

    // Group anagrams from a list of strings.Its solve method takes a vector of strings, groups them into vectors of anagrams, and returns the result as a vector of these grouped anagrams.
    class GroupAnagram
    {
    public:
        vector<vector<string>> solve(vector<string> strs)
        {
            vector<vector<string>> res;
            unordered_map<string, vector<string>> mp;
            for (const string &str : strs)
            {
                auto s = str;
                sort(s.begin(), s.end());
                mp[s].push_back(str);
            }
            for (const auto &[_, group] : mp)
                res.push_back(group);
            return res;
        }
    };

    // validate whether a given string containing parentheses, braces, and brackets is balanced.It includes a solve method that uses a stack - based approach to ensure every opening symbol has a corresponding and correctly ordered closing symbol.
    class ValidateParenthesis
    {
    public:
        bool solve(string s)
        {
            stack<char> st;

            for (char ch : s)
            {
                if (ch == '(' || ch == '{' || ch == '[')
                    st.push(ch);
                else if (st.size() && ((ch == ')' && st.top() == '(') || (ch == '}' && st.top() == '{') || (ch == ']' && st.top() == '[')))
                    st.pop();
                else
                    return false;
            }
            return st.empty();
        }
    };

    class Palindrome
    {
    public:
        bool solveWithToString(int n)
        {
            if (n < 0)
                return false;
            string left = to_string(n);
            string right = {left.rbegin(), left.rend()};

            return left == right;
        }
        bool solve(int x)
        {
            if (x < 0)
                return false;
            int left = 0, right = x;

            while (right > 0)
            {
                left = left * 10 + right % 10;
                right /= 10;
            }
            return left == x;
        }
    };

    // computes the longest palindromic substring within a given string.It uses a two - pointer approach to expand around potential palindrome centers, considering both odd - and even - length palindromes.
    class LongestPalindromicSubstring
    {
    public:
        string solve(string s)
        {
            int start = 0, resLen = 1;
            const int len = s.size();

            for (int ind = 0; ind < len; ++ind)
            {
                int l = ind, r = ind;
                while (l >= 0 && r < len && s[l] == s[r])
                {
                    if ((r - l + 1) > resLen)
                    {
                        resLen = r - l + 1;
                        start = l;
                    }
                    --l, ++r;
                }
                l = ind, r = ind + 1;
                while (l >= 0 && r < len && s[l] == s[r])
                {
                    if ((r - l + 1) > resLen)
                    {
                        resLen = r - l + 1;
                        start = l;
                    }
                    --l, ++r;
                }
            }

            return s.substr(start, resLen);
        }
    };

    // Count the number of palindromic substrings in a given string.It includes a private helper method pd to check if a substring is a palindrome and a public method solve that iterates through all possible substrings to compute the total count of palindromic substrings.
    class CountPalindrome
    {
        bool pd(const string &s, int beg, int end)
        {
            while (beg < end)
            {
                if (s[beg] != s[end])
                    return false;
                ++beg, --end;
            }
            return true;
        }

    public:
        int solve(string s)
        {
            int res = 0;

            for (int i = 0; i < s.size(); ++i)
            {
                for (int j = i; j < s.size(); ++j)
                {
                    res += pd(s, i, j);
                }
            }
            return res;
        }
    };

    class EncodeDecode
    {
        vector<pair<int, int>> vp;

    public:
        string encode(vector<string> arr)
        {
            string s;
            for (const auto &a : arr)
            {
                vp.push_back({s.size(), a.size()});
            }
            return s;
        }
        vector<string> decode(string s)
        {
            vector<string> res;

            for (auto [start, len] : vp)
            {
                res.push_back(s.substr(start, len));
            }

            return res;
        }
    };
} // string

namespace Tree
{
    // Definition for a binary tree node.
    struct TreeNode
    {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode() : val(0), left(nullptr), right(nullptr) {}
        TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
        TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
    };

    // calculate the maximum depth of a binary tree.The solve function uses recursion to traverse the tree and determine the depth by comparing the depths of the left and right subtrees.
    class MaxDepth
    {
    public:
        int solve(TreeNode *root)
        {
            if (!root)
                return 0;
            return 1 + max(solve(root->left), solve(root->right));
        }
    };

    // Determine whether two binary trees are structurally identical and have the same node values. It includes a solve method that recursively compares the nodes of two trees to check for equality.
    class SameTree
    {
    public:
        bool solve(TreeNode *a, TreeNode *b)
        {
            if (!a)
                return b == nullptr;
            if (!b || a->val != b->val)
                return false;
            return solve(a->left, b->left) && solve(b->right, a->right);
        }
    };

    // Recursively swaps the left and right child nodes of each subtree, starting from the root, and returns the modified tree.
    class InvertTree
    {
    public:
        TreeNode *invert(TreeNode *root)
        {
            if (root)
            {
                swap(root->left, root->right);
                invert(root->left);
                invert(root->right);
            }
            return root;
        };
    }; // tree

    class LevelOrder {
        public:
        VI2 solve(TreeNode* root) {
            if(!root) return {{}};
            VI2 res;
            queue<TreeNode*> q;
            q.push(root);

            while(q.size()) {
                int len = q.size();
                vector<int> c;
                while(len--) {
                    auto* t = q.front();
                    c.push_back(t->val);
                    q.pop();
                    if(t->left) q.push(t->left);
                    if(t->right) q.push(t->right);
                }
                res.push_back(c);
            }
            return res;
        }
    };
}
int main()
{
    LinkedList::Node *head = nullptr;
    LinkedList::Reverse rev;
    rev.reverseLoop(head);

    cout << "\n\n*************** Exiting the  main function -------------------";
    return 0;
}