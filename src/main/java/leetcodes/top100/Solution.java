package leetcodes.top100;

import java.util.Arrays;
import java.util.List;

/**
 * @author 张壮
 * @description leetcode top100 https://leetcode.cn/studyplan/top-100-liked/
 * @since 2023/10/14 16:51
 **/
public class Solution {
    public static void main(String[] args) {


        Solution solution = new Solution();
//        solution.nextPermutation(new int[]{1, 3, 4, 5});
//        solution.majorityElement(new int[]{7, 7, 5, 7, 5, 1, 5, 7, 5, 5, 7, 7, 7, 7, 7, 7});\
//        solution.minDistance("bvdab","pandans");
//        solution.longestCommonSubsequence("padadad","pbaadcnndas");
//        solution.longestPalindrome("bacddcab");
//        solution.minPathSum(new int[][]{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}});
//        solution.uniquePaths(3,7);
//        solution.longestValidParentheses2(")()())");
//        solution.canPartition(new int[]{2,4,5,9,15,3,2});
//        solution.maxProduct(new int[]{2,3,4,-2,3,2});
//        solution.lengthOfLIS2(new int[]{10, 9, 2, 5, 3, 7, 101, 18});
//        solution.wordBreak("leetcode", Arrays.asList("leet", "co", "ode", "de"));
//        solution.coinChange(new int[]{1,2,5},11);
//        solution.numSquares(12);
//        solution.rob(new int[]{2, 7, 9, -3, 1, 12});
//        solution.climbStairs(2);
        System.out.println(solution.canPartitionKSubsets(new int[]{1, 5, 2, 9, 5, 3, 8}, 3));
    }


    //额外题目!!!!
    // 698. 分为k个相等的子集 方法一：桶选球，复杂度O（2^n）^k
    //todo
    boolean[] used;
    public boolean canPartitionKSubsets(int[] nums, int k) {
        Arrays.sort(nums);
        used = new boolean[nums.length];
        int sum = Arrays.stream(nums).sum();
        if (sum % k != 0) {
            return false;
        }
        int target = sum / k;
        System.out.println(target);
        if (nums[nums.length - 1] > target) return false;
        return fun2(nums, k, target, 0, nums.length - 1);

    }


    private boolean fun2(int[] nums, int k, int target, int temp, int index) {

        if (k == 1) {

            return true;
        }
        if (temp == target) {
            return fun2(nums, k - 1, target, 0, nums.length - 1);

        }


        for (int i = index; i >= 0; i--) {
            if (used[i] || nums[i] + temp > target) continue;
            used[i] = true;
            System.out.println("选择"+nums[i]);
            if (fun2(nums, k, target, temp + nums[i], index - 1)) {

                return true;
            }
            used[i] = false;
            while (i > 0 && nums[i] == nums[i - 1]) i--;
        }
        return false;

    }



    // 70. 爬楼梯
    public int climbStairs(int n) {

        int dp[] = new int[n + 1];
        dp[1] = 1;
        dp[0] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        return dp[n];

    }

    // 198. 打家劫舍  空间复杂度O(n),时间复杂度O(n)
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];

        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(dp[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        System.out.println(Arrays.toString(dp));

        return dp[nums.length - 1];
    }

    // 198. 打家劫舍 法二 空间复杂度O(1),时间复杂度O(n)
    public int rob2(int[] nums) {
        int pre_2 = 0;
        int pre_1 = 0;
        for (int num : nums) {
            int temp = Math.max(num + pre_2, pre_1);
            pre_2 = pre_1;
            pre_1 = temp;

        }
        return pre_1;
    }

    // 279. 完全平方数
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
            System.out.println("i = " + i);
            for (int j = 1; j * j <= i; j++) {
                System.out.printf("dp[%d-%d] + 1 = %d ,dp[%d] =%d \n", i, j * j, dp[i - j * j] + 1, i, dp[i]);
                dp[i] = Math.min(dp[i - j * j] + 1, dp[i]);
            }

        }
        System.out.println(Arrays.toString(dp));
        return dp[n];
    }

    //322. 零钱兑换
    public int coinChange(int[] coins, int amount) {

        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (coins[j] <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        System.out.println(Arrays.toString(dp));
        return dp[amount] > amount ? -1 : dp[amount];
    }

    // 139. 单词拆分  动态规划
    public boolean wordBreak(String s, List<String> wordDict) {
        //dp[i]表示长度为i的情况下，能否由wordDict拼出。
        boolean[] dp = new boolean[s.length() + 1];

        Arrays.fill(dp, false);
        dp[0] = true;

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                String subString = s.substring(j, i);
                if (wordDict.contains(subString) && dp[j]) {
                    System.out.println("wordDict中包含" + subString);
                    dp[i] = true;
                    break;
                }
            }
        }
        System.out.println(Arrays.toString(dp));
        return dp[s.length()];


    }

    //300. 最长递增子序列  动态规划时间复杂度O(N^2)
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int ans = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            ans = Math.max(ans, dp[i]);


        }
        return ans;
    }

    // 300. 最长递增子序列  二分查找+动态规划   时间复杂度O(NlgN)
    public int lengthOfLIS2(int[] nums) {
        int[] tails = new int[nums.length]; //表示长度为i+1的子序列尾部最小值,例如tails[199] = 299,当严格递增序列长度为199时,尾部最小值为299
        int res = 0;
        for (int cur : nums) {
            int i = 0, j = res;

            System.out.println("i = " + i + ", res = " + res);
            //确定当前数 的位置,如果它比tail[res-1]还大,那么就更新tail[res-1]的值
            while (i < j) {
                int m = (i + j) / 2;
                System.out.println("cur = " + cur + " ,m = " + m);
                if (tails[m] < cur) {
                    i = m + 1;
                } else {
                    j = m;
                }
            }
            tails[i] = cur;
            System.out.println(Arrays.toString(tails));
            if (res == i) res++;
        }
        return res;
    }

    // 152 乘积最大子数组
    public int maxProduct(int[] nums) {
        int max = 1;
        int min = 1;
        int ans = Integer.MIN_VALUE;
        for (int num : nums) {

            if (num < 0) {
                min = min ^ max;
                max = min ^ max;
                min = min ^ max;
            }

            max = Math.max(num, num * max);
            min = Math.min(num, num * min);
            ans = Math.max(max, ans);
        }
        return ans;

    }

    // 152 乘积最大子数组 - 动态规划
    public int maxProduct2(int[] nums) {
        int length = nums.length;
        int[] maxF = new int[length];
        int[] minF = new int[length];
        System.arraycopy(nums, 0, maxF, 0, length);
        System.arraycopy(nums, 0, minF, 0, length);
        for (int i = 1; i < length; ++i) {
            maxF[i] = Math.max(maxF[i - 1] * nums[i], Math.max(nums[i], minF[i - 1] * nums[i]));
            minF[i] = Math.min(minF[i - 1] * nums[i], Math.min(nums[i], maxF[i - 1] * nums[i]));
        }
        int ans = maxF[0];
        for (int i = 1; i < length; ++i) {
            ans = Math.max(ans, maxF[i]);
        }
        return ans;
    }


    //416. 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).parallel().sum();
        if (sum % 2 != 0) return false;
        int target = sum / 2;

        int[] dp = new int[target + 1];
        for (int i = 0; i < nums.length - 1; i++) {
            System.out.println("--当前物品为：" + nums[i]);
            for (int j = target; j >= nums[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - nums[i]] + nums[i]);
                System.out.print("dp[" + j + "] = " + dp[j] + ", ");
            }
            System.out.println();
        }
        System.out.println(Arrays.toString(dp));

        return dp[target] == target;
    }

    // 32. 最长有效括号 方法二 : 动态规划
    public int longestValidParentheses2(String s) {
        //dp[i] 表示 [0,i]区间里以第i个字符为结尾的有效括号数量
        int dp[] = new int[s.length()];
        Arrays.fill(dp, 0);

        char[] chars = s.toCharArray();

        int maxLen = 0;
        //i从第2个开始,因为有效括号必须为偶数
        for (int R = 1; R < chars.length; R++) {
            if (chars[R] == ')') {
                int L = R - dp[R - 1] - 1;
                System.out.println("R = " + R + ",L = " + L);
                if (L >= 0 && chars[L] == '(')
                    dp[R] = 2 + ((L - 1 >= 0) ? dp[L - 1] : 0) + dp[R - 1];

            }
            maxLen = Math.max(maxLen, dp[R]);
        }
        System.out.println(Arrays.toString(dp));
        return maxLen;
    }

    //62. 不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];

        for (int i = 0; i < n; i++)
            dp[0][i] = 1;
        for (int j = 0; j < m; j++)
            dp[j][0] = 1;

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] += (dp[i][j - 1] + dp[i - 1][j]);
            }
        }
        System.out.println("dp数组如下：");
        for (int i = 0; i < m; i++) {
            System.out.println(Arrays.toString(dp[i]));
        }
        return dp[m - 1][n - 1];
    }

    //64. 最小路径和
    public int minPathSum(int[][] grid) {
        int colLen = grid.length;
        int rowLen = grid[0].length;

        int[][] dp = new int[colLen][rowLen];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < colLen; i++)
            dp[0][i] = grid[0][i] + dp[0][i - 1];
        for (int i = 1; i < rowLen; i++)
            dp[i][0] = grid[i][0] + dp[i - 1][0];

        for (int i = 1; i < rowLen; i++)
            for (int j = 1; j < colLen; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        System.out.println("dp数组如下：");
        for (int i = 0; i < colLen; i++) {
            System.out.println(Arrays.toString(dp[i]));
        }
        return dp[rowLen - 1][colLen - 1];

    }

    // 5.最长回文子串 动态规划
    public String longestPalindrome(String s) {
        int len = s.length();
        boolean dp[][] = new boolean[len][len];
        for (int i = 0; i < len; i++)
            dp[i][i] = true;
        int maxLen = 1;
        int l = 0;
        char arr[] = s.toCharArray();
        for (int step = 2; step <= len; step++) {
            for (int L = 0; L < len; L++) {
                int R = L + step - 1;
                if (R >= len) break;
                if (arr[L] != arr[R]) dp[L][R] = false;
                else {//arr[L] == arr[R]
                    if (R == L + 1 || R == L + 2) dp[L][R] = true;
                    dp[L][R] = dp[L + 1][R - 1];
                }

                if (dp[L][R] && step > maxLen) {
                    maxLen = step;
                    l = L;
                }
            }
        }
        System.out.println("dp数组如下：");
        for (int i = 0; i < len; i++) {
            System.out.println(Arrays.toString(dp[i]));
        }
        return s.substring(l, l + maxLen);
    }

    // 1143.最长公共子序列（LCS）
    public int longestCommonSubsequence(String text1, String text2) {
        int len1 = text1.length();
        int len2 = text2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                } else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }

        }
        System.out.println("dp数组如下：");
        for (int i = 0; i <= len1; i++) {
            System.out.println(Arrays.toString(dp[i]));
        }

        return dp[len1][len2];
    }

    //72.编辑距离
    public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();

        int[][] dp = new int[len1 + 1][1 + len2];//dp[i][j]标识word1[0,i-1]于word2[0,j-1]的最小步数

        for (int i = 0; i <= len1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= len2; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    int temp = Math.min(dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1);
                    dp[i][j] = Math.min(temp, dp[i][j - 1] + 1);
                }
            }
        }
        System.out.println("dp数组如下：");
        for (int i = 0; i <= len1; i++) {
            System.out.println(Arrays.toString(dp[i]));
        }

        return dp[len1][len2];
    }

    // 136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        int e = 0;
        for (int num : nums) {
            e ^= num;
        }
        return e;

    }

    // 169. 多数元素
    // 思想: 众数到最后一定比其它数多,每次遍历记录某一个数X的出现次数,遇到不相同的就-1,直到0,更换其它众数Y
    public int majorityElement(int[] nums) {
        int ans = 0;
        int count = 0;
        for (int num : nums) {
            System.out.print("ans = " + ans + ",当前 num =" + num);
            if (num == ans) {
                count++;
            } else if (count == 0) {
                ans = num;
                count = 1;
            } else count--;
            System.out.println(",ans = " + ans + ",当前 count = " + count);

        }
        return ans;
    }

    //75. 颜色分类
    public void sortColors(int[] nums) {
        int end_index_0 = -1;
        int start_index_2 = nums.length;
        int i = 0;
        while (i < start_index_2) {
            if (nums[i] == 1) i++;
            else if (nums[i] == 0) {
                swap(nums, i, ++end_index_0);
                i++;
            } else swap(nums, i, --start_index_2);
        }
    }

    public void swap(int[] nums, int l, int r) {
        int temp = nums[l];
        nums[l] = nums[r];
        nums[r] = temp;
    }

    //75. 颜色分类  - 计数排序法
    public void sortColors_2(int[] nums) {
        int[] cnt = new int[3];
        for (int num : nums) {
            cnt[num]++;
        }

        for (int i = 0; i < cnt[0]; i++) {
            nums[i] = 0;
        }
        for (int i = cnt[0]; i < cnt[0] + cnt[1]; i++) {
            nums[i] = 1;
        }
        for (int i = cnt[0] + cnt[1]; i < cnt[0] + cnt[1] + cnt[2]; i++) {
            nums[i] = 2;
        }


    }

    //287. 寻找重复数
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        do {
            slow = nums[0];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    // 31. 下一个排列
    public void nextPermutation(int[] nums) {
        int len = nums.length;
        int right = 0, left = 0;
        //从后往前找,第一个不符合升序的元素nums[right-1],然后在[right,len)中找到比它大的最小数
        for (right = len - 1; right >= 1; right--) {
            if (nums[right - 1] < nums[right]) {
                left = right - 1;
                System.out.println(nums[right - 1] + "比" + nums[right] + "小：在[" + left + ",len)中找到比 nums[" + left + "]=" + nums[left] + "小的数进行交换");
                // 在 [right,len-1]中找到一个比 nums[left] 大的数,进行交换
                for (int i = len - 1; i >= right; i--) {
                    if (nums[i] > nums[left]) {
                        int temp = nums[i];
                        nums[i] = nums[left];
                        nums[left] = temp;
                        break;
                    }
                }
                // 将后面部分排序
                Arrays.sort(nums, right, len);
                break;
            }

        }

        // 特殊情况: 数组原本是逆序排列,那么下一个序列应该为正序的数组
        if (right == 0) Arrays.sort(nums);
    }
}

