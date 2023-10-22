package leetcodes;

import data.structure.ListNode;
import data.structure.TreeNode;

import cn.hutool.core.util.NumberUtil;
import lombok.extern.slf4j.Slf4j;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.IntStream;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/2/2 18:31
 **/
@Slf4j
public class Solution {
    //测试用的主方法
    public static void main(String[] args) {
//        Solution solution = new Solution();
//
//
//        ListNode root = new ListNode(1);
//        ListNode node2 = new ListNode(2);
//        ListNode node3 = new ListNode(3);
//        ListNode node4 = new ListNode(4);
//        ListNode node5 = new ListNode(5);
//        root.next = node2;
//        node2.next = node5;
//        node5.next = node4;
//        node4.next = node3;
////        log.info(root.toString());
////        ListNode listNode = solution.sortList(root);
////        log.info(listNode.toString());
//
////        List<List<String>> partition = solution.partition("abbab");
////        for (List cur : partition) {
////            log.info(cur.toString());
////        }
//
//        List<String> strings = solution.restoreIpAddresses("25525511135");

//        System.out.println(strings);

        BigDecimal bigDecimal = new BigDecimal("4.44");
        BigDecimal decimal = new BigDecimal(4.44);
        BigDecimal valueOf = BigDecimal.valueOf(4.44);
        BigDecimal bigDecimal2 = new BigDecimal(999999999999L);
        BigDecimal valueOf2 = BigDecimal.valueOf(999999999999L);
        System.err.println("bigDecimal=" + bigDecimal);
        System.err.println("decimal=" + decimal);
        System.err.println("valueOf=" + valueOf);
        System.err.println("bigDecimal2=" + bigDecimal2);
        System.err.println("valueOf2=" + valueOf2);
        double double_param = 2e2;
        System.out.println(double_param);
        HashMap<Integer, Integer> hashMap_param = new HashMap();
        hashMap_param.put(null, 3);
        hashMap_param.put(null, 4);

    }

    //动态规划章节：


    //674. 最长连续递增序列
    public int findLengthOfLCIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int ans = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) dp[i] = dp[i - 1] + 1;
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }

    //718. 最长重复子数组
    public int findLength(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int[][] dp = new int[len1 + 1][len2 + 1];
        dp[0][0] = 0;
        int ans = 0;

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
                ans = Math.max(ans, dp[i][j]);
            }
        }
        return ans;
    }


    //1035.不相交的线
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int[][] dp = new int[len1 + 1][len2 + 1];

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[len1][len2];
    }

    //53. 最大子数组和
    public int maxSubArray(int[] nums) {

        int len = nums.length;

        if (len == 0) return -1;
        //dp[i]表示[0,i]包含nums【i】的最大子数组和
        int[] dp = new int[len];
        dp[0] = nums[0];
        int max = nums[0];
        for (int i = 1; i < len; i++) {
            dp[i] = Math.max(nums[i], dp[i - 1] + nums[i]);
            max = Math.max(dp[i], max);
        }
        return max;

    }


    //392.判断子序列
    public boolean isSubsequence(String s, String t) {
        int len_s = s.length();
        int len_t = t.length();
        if (len_s > len_t) return false;
        //dp[i][j]表示 s中[0,i）和t[0,j）中相同子序列的最大长度
        int[][] dp = new int[len_s + 1][len_t + 1];
        for (int s_endIndex = 1; s_endIndex <= len_s; s_endIndex++) {
            for (int t_endIndex = 1; t_endIndex <= len_t; t_endIndex++) {
                if (s.charAt(s_endIndex - 1) == t.charAt(t_endIndex - 1))
                    dp[s_endIndex][t_endIndex] = dp[s_endIndex - 1][t_endIndex - 1] + 1;
                else dp[s_endIndex][t_endIndex] = dp[s_endIndex][t_endIndex - 1];
            }
        }
        return dp[len_s][len_t] == len_s;
    }

    //115.不同的子序列
    public int numDistinct(String s, String t) {
        int len_s = s.length();
        int len_t = t.length();
        if (len_s < len_t) return 0;
        //以i-1为结尾的s子序列中出现以j-1为结尾的t的个数为dp[i][j]。
        int[][] dp = new int[len_s + 1][len_t + 1];
        for (int i = 0; i <= len_s; i++)
            dp[i][0] = 1;

        for (int i = 1; i <= len_s; i++) {
            for (int j = 1; j <= len_t; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                else dp[i][j] = dp[i - 1][j];
            }
        }
        return dp[len_s][len_t];
    }

    //583. 两个字符串的删除操作
    public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();
        //dp[i][j]：以i-1为结尾的字符串word1，和以j-1位结尾的字符串word2，想要达到相等，所需要删除元素的最少次数。
        int[][] dp = new int[len1 + 1][1 + len2];//dp[i][j]标识word1[0,i-1]于word2[0,j-1]的最小步数

        for (int i = 0; i < len1 + 1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < len2 + 1; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i < len1 + 1; i++) {
            for (int j = 1; j < len2 + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
                }
            }
        }

        return dp[len1][len2];
    }

    //72.编辑距离
    public int minDistance21(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();

        int[][] dp = new int[len1 + 1][1 + len2];//dp[i][j]标识word1[0,i-1]于word2[0,j-1]的最小步数

        for (int i = 0; i < len1 + 1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < len2 + 1; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i < len1 + 1; i++) {
            for (int j = 1; j < len2 + 1; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    int temp = Math.min(dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1);
                    dp[i][j] = Math.min(temp, dp[i][j - 1] + 1);
                }
            }
        }

        return dp[len1][len2];
    }

    //72. 编辑距离 递归
    int[][] memo;

    public int minDistance22(String word1, String word2) {

        int len1 = word1.length();
        int len2 = word2.length();
        memo = new int[len1][len2];
        return fun(word1, word2, len1 - 1, len2 - 1);
    }

    private int fun(String word1, String word2, int index1, int index2) {
//        char, byte, short, int, Character, Byte, Short, Integer, String, or an enum

        if (index1 == -1 || index2 == -1) {
            return Math.max(index2, index1) + 1;
        }
        if (memo[index1][index2] != 0) {
            return memo[index1][index2];
        }
        if (word1.charAt(index1) == word2.charAt(index2)) {
            memo[index1][index2] = fun(word1, word2, index1 - 1, index2 - 1);
            return memo[index1][index2];
        }


        memo[index1][index2] = 0;
        int temp = Math.min(fun(word1, word2, index1, index2 - 1), fun(word1, word2, index1 - 1, index2 - 1));
        memo[index1][index2] = 1 + Math.min(temp, fun(word1, word2, index1 - 1, index2));
        return memo[index1][index2];
    }

    // 647. 回文子串  -  返回字符串中回文子串的个数，动态规划
    public int countSubstrings(String s) {
        int len = s.length();
        int res = 0;
        boolean[][] dp = new boolean[len][len];
        //从最后一个元素开始
        for (int L = len - 1; L >= 0; L--) {
            for (int R = L; R < len; R++) {
                if (s.charAt(L) == s.charAt(R) && (R <= 1 + L || dp[L + 1][R - 1])) {
                    res++;
                    dp[L][R] = true;

                }

            }
        }
        return res;

    }

    // 647. 回文子串  -  返回字符串中回文子串的个数，动态规划
    public int countSubstrings2(String s) {
        int ans = 0;
        for (int i = 0; i < s.length(); i++) {
            ans += extend(s, i, i, s.length());
            ans += extend(s, i, i + 1, s.length());
        }
        return ans;
    }

    private int extend(String s, int L, int R, int n) {
        if (L >= s.length()) return 0;
        int ans = 0;
        while (L >= 0 && R < n && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
            ans++;
        }
        return ans;
    }

    //516.最长回文子序列
    public int longestPalindromeSubseq(String s) {
        int len = s.length();
        int maxLen = 1;
        int[][] dp = new int[len][len];
        for (int i = 0; i < len; i++) {
            dp[i][i] = 1;
        }
        for (int L = len - 1; L >= 0; L--) {
            for (int R = L + 1; R < len; R++) {
                if (s.charAt(R) == s.charAt(L)) {
                    dp[L][R] = 2 + dp[L + 1][R - 1];

                } else {
                    dp[L][R] = Math.max(dp[L][R - 1], dp[L + 1][R]);
                }
                if (dp[L][R] > maxLen) {
                    maxLen = dp[L][R];
                }

            }
        }
        return maxLen;
    }


    //121. 买卖股票的最佳时机
    public int maxProfit_121(int[] prices) {
        if (prices == null || prices.length == 0) return 0;
        int[][] dp = new int[prices.length][2];
        //dp[i][0]表示第i天持有股票获得最大收益
        //dp[i][1]表示第i天不持有股票获得最大收益
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            dp[i][1] = Math.max(dp[i - 1][0] + prices[i], dp[i - 1][1]);

        }
        return dp[prices.length - 1][1];
    }

    //122.买卖股票的最佳时机II
    public int maxProfit_122(int[] prices) {
        if (prices == null || prices.length == 0) return 0;
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        return dp[prices.length - 1][1];
    }

    //123.买卖股票的最佳时机III
    public int maxProfit_123(int[] prices) {
        int len = prices.length;
        if (len == 0) return 0;
        int[][] dp = new int[len][5];
        //dp[i][1]表示第i天，第一次持有股票
        //dp[i][2]表示第i天，第一次不持有股票
        //dp[i][3]表示第i天，第二次持有股票
        //dp[i][4]表示第i天，第二次不持有股票
        dp[0][1] = -prices[0];
        dp[0][2] = 0;
        dp[0][3] = -prices[0];
        dp[0][4] = 0;
        for (int i = 1; i < len; i++) {
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
        }
        return dp[len - 1][4];
    }

    //188.买卖股票的最佳时机IV
    public int maxProfit_188(int[] prices, int k) {
        if (prices == null || prices.length == 0 || k <= 0) return 0;
        int[][] dp = new int[prices.length][k * 2 + 1];//奇书表水手里有股票最高收益，偶数表示当前手里没票时最高收益
        dp[0][0] = 0;
        for (int i = 0; i < 2 * k + 1; i++) {
            if (i % 2 != 0) dp[0][i] = -prices[0];
            else dp[0][i] = 0;
        }

        for (int i = 1; i < prices.length; i++) {
            for (int j = 0; j < 2 * k - 1; j += 2) {
                dp[i][j + 1] = Math.max(dp[i - 1][j + 1], dp[i - 1][j] - prices[i]); //持有
                dp[i][j + 2] = Math.max(dp[i - 1][j + 2], dp[i - 1][j + 1] + prices[i]);
            }
        }
        return dp[prices.length - 1][2 * k];
    }

    //309.最佳买卖股票时机含冷冻期
    public int maxProfit_309(int[] prices) {
        if (prices == null || prices.length == 0 || prices.length == 1) return 0;
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];
        dp[1][0] = Math.max(dp[0][0], -prices[1]);
        dp[1][1] = Math.max(0, dp[0][0] + prices[1]);
        for (int i = 2; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 2][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        return dp[prices.length - 1][1];
    }

    //714.买卖股票的最佳时机含手续费
    public int maxProfit_714(int[] prices, int fee) {
        if (prices == null || prices.length == 0) return 0;

        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];
        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i] - fee);
        }
        return dp[prices.length - 1][1];
    }

    //↑↑↑↑↑ -- 动态规划--↑↑↑↑↑↑↑

    //93.复原IP地址
    public List<String> restoreIpAddresses(String s) {
        List<String> ans = new ArrayList<>();
        fun_93(s, 0, 0, ans, new LinkedList<>());
        return ans;

    }

    private void fun_93(String s, int index, int pointSum, List<String> ans, LinkedList<String> cur) {
        System.out.println(cur);
        if (pointSum == 4 && s.length() == index) {

            StringBuilder sb = new StringBuilder();
            for (String curString : cur) {
                sb.append(curString);
            }
            sb.deleteCharAt(sb.length() - 1);
            ans.add(sb.toString());
            return;
        }
        if (pointSum == 4 && index != s.length()) return;
        for (int i = index; i < s.length() && i < index + 3; i++) {
            if (isValid(s, index, i)) {
                cur.add(s.substring(index, i + 1) + ".");
                fun_93(s, i + 1, pointSum + 1, ans, cur);
                cur.removeLast();
            }

        }
    }

    private static boolean isValid(String s, int L, int R) {
        if (R == L) return true;
        if (s.charAt(L) == '0') return false;
        String sb = s.substring(L, R + 1);
        if (Integer.valueOf(sb) > 255) return false;
        return true;
    }

    //125.验证回文串
    public boolean isPalindrome(String s) {
        int n = s.length();
        int left = 0, right = n - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                ++left;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                --right;
            }
            if (left < right) {
                if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                    return false;
                }
                ++left;
                --right;
            }
        }
        return true;
    }

    //131.分割回文串、
    boolean[][] dp;
    List<List<String>> ans_131 = new ArrayList<>();
    LinkedList<String> cur = new LinkedList<>();

    public List<List<String>> partition(String s) {
        dp = new boolean[s.length()][s.length()];
        for (int L = s.length() - 1; L >= 0; L--) {
            for (int R = L; R < s.length(); R++) {
                if (s.charAt(L) == s.charAt(R)) {
                    if (R - L <= 2 || dp[L + 1][R - 1]) dp[L][R] = true;
                }
            }
        }
        fun(s, 0);
        return ans_131;
    }

    private void fun(String s, int i) {
        if (i == s.length()) {
            ans_131.add(new ArrayList<>(cur));
        }
        for (int R = i; R < s.length(); R++) {
            if (dp[i][R] == true) {
                cur.add(s.substring(i, R + 1));

                fun(s, R + 1);
                cur.removeLast();
            }
        }
    }

    //1005.K次取反后最大化的数组和
    public int largestSumAfterKNegations(int[] nums, int k) {
        nums = IntStream.of(nums).boxed().sorted((o1, o2) -> Math.abs(o2) - Math.abs(o1)).mapToInt(Integer::intValue).toArray();
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0 && k >= 1) {
                k--;
                nums[i] = -nums[i];
            }
        }
        if (k % 2 == 1) nums[nums.length - 1] = -nums[nums.length - 1];
        return Arrays.stream(nums).sum();
    }


    //496.下一个更大的元素
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
//       1.栈顶最小，栈底最大
        LinkedList<Integer> max_stack = new LinkedList();//
        HashMap<Integer, Integer> map = new HashMap();
        int[] ans = new int[nums1.length];
        Arrays.fill(ans, -1);
        for (int i = 0; i < nums1.length; i++) {
            map.put(nums1[i], i);
        }
        max_stack.push(0);
        for (int i = 0; i < nums2.length; i++) {
            while (!max_stack.isEmpty() && nums2[i] > nums2[max_stack.peek()]) {
                if (map.containsKey(nums2[max_stack.peek()])) {
                    ans[map.get(nums2[max_stack.peek()])] = nums2[i];
                }

                max_stack.pop();
            }
            max_stack.push(i);
        }
        return ans;
    }

    //503.下一个更大元素 2.0
    public int[] nextGreaterElements(int[] nums) {
        //边界判断
        if (nums == null || nums.length <= 1) {
            return new int[]{-1};
        }
        int size = nums.length;
        int[] result = new int[size];//存放结果
        Arrays.fill(result, -1);//默认全部初始化为-1
        LinkedList<Integer> min_max_stack = new LinkedList<>();//栈中存放的是nums中的元素下标
        for (int i = 0; i < 2 * size; i++) {
            while (!min_max_stack.isEmpty() && nums[i % size] > nums[min_max_stack.peek()]) {
                result[min_max_stack.peek()] = nums[i % size];//更新result
                min_max_stack.pop();//弹出栈顶
            }
            min_max_stack.push(i % size);
        }
        return result;
    }

    //42. 接雨水
    public int trap(int[] height) {
        int len = height.length;
        int leftMax[] = new int[len];
        leftMax[0] = height[0];
        int rightMax[] = new int[len];
        rightMax[len - 1] = height[len - 1];
        //[0,i] 最大的值
        for (int i = 1; i < len; i++) {
            leftMax[i] = Math.max(height[i], leftMax[i - 1]);
        }
        //[i,len-1]最大值
        for (int i = len - 2; i >= 0; i--) {
            rightMax[i] = Math.max(height[i], rightMax[i + 1]);
        }

        int ans = 0;
        for (int i = 0; i < len; i++) {
            ans += (Math.min(leftMax[i], rightMax[i]) - height[i]) * 1;

        }
        return ans;


    }


    //232
    class MyQueue {
        LinkedList<Integer> stack_in;
        LinkedList<Integer> stack_out;

        public MyQueue() {
            stack_in = new LinkedList<>();
            stack_out = new LinkedList<>();
        }

        public void push(int x) {
            stack_in.push(x);
        }

        public int pop() {
            if (stack_out.isEmpty() == false) return stack_out.pop();
            while (stack_in.isEmpty() == false) {
                stack_out.push(stack_in.pop());
            }
            if (stack_out.isEmpty() == false) return stack_out.pop();
            return -1;
        }

        public int peek() {
            if (stack_out.isEmpty() == false) return stack_out.peek();
            while (stack_in.isEmpty() == false) {
                stack_out.push(stack_in.pop());
            }
            if (stack_out.isEmpty() == false) return stack_out.peek();
            return -1;
        }

        public boolean empty() {
            return stack_out.isEmpty() && stack_in.isEmpty();
        }
    }


    //76
    public String minWindow(String s, String t) {
        String ansString = "";
        int[] t_table = new int[123];
        for (int i = 0; i < t.length(); i++) {
            t_table[t.charAt(i)]++;
        }

        int l = 0, r = 0;
        int rest = t.length();
        int ans = Integer.MAX_VALUE;
        while (r < s.length()) {
            if (t_table[s.charAt(r)] >= 1) rest--;
            t_table[s.charAt(r)]--;

            //此时统计窗口中包含t的子串长度
            if (rest == 0) {
                while (l < r && t_table[s.charAt(l)] < 0) {
                    t_table[s.charAt(l)]++;
                    l++;
                }
                if (ans > r - l + 1) {
                    ans = r - l + 1;
                    ansString = s.substring(l, l + ans);
                }
                t_table[s.charAt(l)]++;
                l++;
                rest++;
            }
            r++;
        }
        return ansString;

    }

    //1135
    public int[] smallerNumbersThanCurrent(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap();
        Arrays.sort(nums);
        int[] ans = new int[nums.length];
        for (int i = nums.length - 1; i > -1; i--) {
            map.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            ans[i] = map.get(nums[i]);
        }
        System.out.println(Arrays.toString(ans));
        return ans;

    }

    // 1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length < 2) return null;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return null;
    }

    // 2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        if (l1 == null || l2 == null) return l1 == null ? l2 : l1;

        ListNode ans = new ListNode(0);

        ListNode p = l1;
        ListNode q = l2;
        ListNode cur = ans;
        int temp = 0;
        int remainder = 0;
        //当 pq都走到头时退出
        while (p != null || q != null) {
            int v1 = p == null ? 0 : p.val;
            int v2 = q == null ? 0 : q.val;
            remainder = (v1 + v2 + temp) % 10;
            temp = (v1 + v2 + temp) / 10;

            ListNode node = new ListNode(remainder);
            cur.next = node;
            cur = cur.next;
            p = (p == null) ? null : p.next;
            q = (q == null) ? null : q.next;
        }
        if (temp == 1) cur.next = new ListNode(1);
        return ans.next;
    }

    // 3 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {

        if (s == null || s == "" || s.length() == 0) return 0;
        int l = 0;
        int r = 0;
        int curMaxLength = 0;
        // map 记录遍历时字符出现的最新位置.
        Map<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            r = i;
            Character c = s.charAt(i);
            if (map.containsKey(c)) {
                // 一旦窗口中有重复的内容出现,就修改左边界l,将l置为 map中该字符的位置的下一位.
                l = Math.max(l, map.get(c) + 1);

            }
            map.put(c, i);
            curMaxLength = Math.max(curMaxLength, r - l + 1);
        }
        return curMaxLength;
    }


    // 4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {

        int length1 = nums1.length;
        int length2 = nums2.length;
        if (NumberUtil.isOdd(length1 + length2)) {
            int kth = (length1 + length2) / 2 + 1;
            return kthNumber(nums1, nums2, kth);
        } else {
            int temp1 = kthNumber(nums1, nums2, (length2 + length1) / 2);
            int temp2 = kthNumber(nums1, nums2, (length2 + length1) / 2 + 1);
            return (double) (temp1 + temp2) / 2;
        }

    }

    // 返回两个正序数组中第k小的数
    private int kthNumber(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int start1 = 0;
        int start2 = 0;
        while (true) {
            if (start1 == len1) return nums2[start2 + k - 1];
            if (start2 == len2) return nums1[start1 + k - 1];

            if (k == 1) return Math.min(nums1[start1], nums2[start2]);

            int half = k / 2;
            int p = Math.min(len1, start1 + half) - 1;


            int q = Math.min(len2, start2 + half) - 1;

            if (nums1[p] > nums2[q]) {
                k -= (q - start2 + 1);
                start2 = q + 1;
            } else {
                k -= (p - start1 + 1);
                start1 = p + 1;
            }


        }

    }


    //11. 盛最多水的容器
    public int maxArea(int[] height) {
        int L = 0;
        int R = height.length - 1;
        int ans = 0;
        int temp = 0;
        while (L < R) {
            if (height[L] < height[R]) {
                temp = (R - L) * height[L];
                L++;
            } else { //height[L] 大于或等于 height[R]
                temp = (R - L) * height[R];
                R--;
            }
            ans = Math.max(ans, temp);
        }
        return ans;
    }

    // 15. 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();

        int length = nums.length;
        Arrays.sort(nums);
        if (nums[0] > 0) return ans;
        for (int one = 0; one < length; one++) {
            if (one > 0 && nums[one] == nums[one - 1]) continue;
            int temp = -nums[one];
            int three = length - 1;
            for (int two = one + 1; two < length; two++) {
                if (two > one + 1 && nums[two] == nums[two - 1]) continue;

                while (two < three && nums[two] + nums[three] > temp) three--;
                if (two == three) break;
                if (nums[two] + nums[three] == temp) ans.add(Arrays.asList(nums[one], nums[two], nums[three]));
            }
        }


        return ans;
    }

    // 17. 电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList<>();
        Map<Character, String> map = new HashMap<>();
        if (digits.length() == 0) {
            return ans;
        }
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");

        findAns(ans, new StringBuilder(), digits, map, 0);
        return ans;
    }

    private void findAns(List<String> ans, StringBuilder temp, String digits, Map<Character, String> map, int index) {
        if (digits.length() == index) {
            ans.add(temp.toString());
            return;
        }
        String curr = map.get(digits.charAt(index));
        for (int i = 0; i < curr.length(); i++) {
            temp.append(curr.charAt(i));
            findAns(ans, temp, digits, map, index + 1);
            temp.deleteCharAt(index);
        }
    }


    //19. 删除链表的倒数第 N 个结点  方法一 常规思路法
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int length = 0;
        ListNode cur = head;

        while (cur != null) {
            length++;
            cur = cur.next;
        }
        cur = head;

        if (n == length) return cur.next;
        for (int i = 1; i < length - n; i++) {
            cur = cur.next;
        }
        cur.next = cur.next.next;
        return head;
    }


    //19. 删除链表的倒数第 N 个结点  方法二  双指针法
    public ListNode removeNthFromEnd2(ListNode head, int n) {
        ListNode fast = head;
        ListNode slow = head;

        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        //特殊情况:  删除的结点为 头结点
        if (fast == null) return head.next;

        while (fast != null && slow != null) {
            fast = fast.next;

            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }


    // 21. 合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null || list2 == null) return list1 == null ? list1 : list2;

        ListNode cur1 = list1;
        ListNode cur2 = list2;
        ListNode ansHead = new ListNode(0);
        ListNode cur = ansHead;
        while (cur1 != null && cur2 != null) {
            if (cur1.val < cur2.val) {

                cur.next = cur1;
                cur = cur.next;
                cur1 = cur1.next;
            } else {
                cur.next = cur2;
                cur = cur.next;
                cur2 = cur2.next;
            }

        }
        if (cur1 != null) cur.next = cur1;
        if (cur2 != null) cur.next = cur2;
        return ansHead.next;
    }

    // 22. 括号生成
    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        backTrack(ans, cur, 0, 0, n);
        return ans;

    }

    private void backTrack(List<String> ans, StringBuilder cur, int leftSum, int rightSum, int leftMax) {
        if (cur.length() == 2 * leftMax) {
            ans.add(cur.toString());
        }
        // 左边的括号 最多能有 n 个.
        if (leftSum < leftMax) {
            cur.append('(');
            backTrack(ans, cur, leftSum + 1, rightSum, leftMax);
            cur.deleteCharAt(cur.length() - 1);
        }
        // 添加 右边的括号
        if (rightSum < leftSum) {
            cur.append(')');
            backTrack(ans, cur, leftSum, rightSum + 1, leftMax);
            cur.deleteCharAt(cur.length() - 1);
        }
    }

    // 23. 合并K个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode ans = new ListNode(0);
        Queue<ListNode> queue = new PriorityQueue<>((v1, v2) -> v1.val - v2.val);


        //将所有链表的头结点 加入到 优先队列中
        for (int i = 0; i < lists.length; i++) {
            if (lists[i] != null) queue.offer(lists[i]);
        }
        ListNode cur = ans;
        while (!queue.isEmpty()) {
            ListNode poll = queue.poll();
            cur.next = poll;
            cur = cur.next;
            if (poll.next != null) queue.offer(poll.next);
        }
        return ans.next;
//        Thread a = new Thread()

    }


    // 33. 在旋转排序数组中搜索
    public int search(int[] nums, int target) {
        int len = nums.length;
        if (len == 0) return -1;
        if (len == 1) return nums[0] == target ? 0 : -1;

        int l = 0, r = len - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) return mid;
            if (nums[0] < nums[mid]) { // 左边有序
                if (nums[l] < target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target < nums[r]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }

        }
        return -1;
    }

    // 34. 在排序数组中查找等于target的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return new int[]{-1, -1};
        }
        int begin = 0;
        int end = 0;
        //查找左边界 begin 大于target -1 的第一个位置
        begin = binarySearch(nums, target - 1);

        //查找右边界 begin 大于target  的第一个位置

        end = binarySearch(nums, target);

        if (nums[begin] == target) return new int[]{begin, end - 1};
        else return new int[]{-1, -1};
    }

    // 寻找大于target 的第一个的位置
    private int binarySearch(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        int ans = 0;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            //在这里确定 第一个位置的下标
            if (target < nums[mid]) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }

        }
        return ans;
    }

    // 39. 组合总和
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        fun(candidates, target, 0, new ArrayList<>(), ans);
        return ans;
    }

    private void fun(int[] arr, int rest, int index, List<Integer> temp, List<List<Integer>> ans) {
        if (index == arr.length) return;
        if (rest == 0) {
            ans.add(temp);
            return;
        }
        fun(arr, rest, index + 1, temp, ans);
        if (arr[index] <= rest) {
            temp.add(arr[index]);
            fun(arr, rest - arr[index], index + 1, temp, ans);
            temp.remove(temp.size() - 1);//回溯算法的精髓所在

        }
    }


    //463.岛屿的周长
    static int[] dx = {0, 1, 0, -1};
    static int[] dy = {1, 0, -1, 0};

    public int islandPerimeter(int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] == 1) {
                    ans += dfs(i, j, grid, n, m);
                }
            }
        }
        return ans;
    }

    public int dfs(int x, int y, int[][] grid, int n, int m) {
        if (x < 0 || x >= n || y < 0 || y >= m || grid[x][y] == 0) {
            return 1;
        }
        if (grid[x][y] == 2) {
            return 0;
        }
        grid[x][y] = 2;
        int res = 0;
        for (int i = 0; i < 4; ++i) {
            int tx = x + dx[i];
            int ty = y + dy[i];
            res += dfs(tx, ty, grid, n, m);
        }
        return res;
    }

    // 46. 全排列
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        if (len == 0) return null;
        if (len == 1) return new ArrayList<>(new ArrayList(Arrays.asList(nums)));

        List<List<Integer>> ans = new ArrayList<>();
        fun(ans, new ArrayList<>(), nums);
        return ans;
    }

    private void fun(List<List<Integer>> ans, ArrayList<Integer> temp, int[] nums) {
        if (temp.size() == nums.length) {
            //这里注意将新的 ArrayList对象放入 ans中
            ans.add(new ArrayList<>(temp));
//            ans.add(temp);
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (!temp.contains(nums[i])) {
                temp.add(nums[i]);
                fun(ans, temp, nums);
                temp.remove(temp.size() - 1);
            }
        }

    }

    //1124. 表现良好的最长时间段
    public int longestWPI(int[] hours) {
        int n = hours.length;
        int[] s = new int[n + 1];
        Stack<Integer> stk = new Stack<>();
        stk.push(0);
        for (int i = 1; i <= n; i++) {
            s[i] = s[i - 1] + (hours[i - 1] > 8 ? 1 : -1);
            if (s[i] < s[stk.peek()]) {
                stk.push(i);
            }
        }

        int res = 0;
        for (int r = n; r >= 1; r--) {
            while (!stk.isEmpty() && s[stk.peek()] < s[r]) {
                res = Math.max(res, r - stk.pop());
            }
        }
        return res;


    }

    // 238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;

        int[] prefixResult = new int[len];
        int[] lastfixResult = new int[len];

        prefixResult[0] = nums[0];

        for (int i = 1; i <= len - 1; i++) {
            prefixResult[i] = prefixResult[i - 1] * nums[i];
        }

        lastfixResult[len - 1] = nums[len - 1];

        for (int i = len - 2; i >= 0; i--) {
            lastfixResult[i] = lastfixResult[i + 1] * nums[i];
        }
        int[] ans = new int[len];
        ans[0] = lastfixResult[1];
        ans[len - 1] = prefixResult[len - 2];
        for (int i = 1; i <= len - 2; i++) {
            ans[i] = prefixResult[i - 1] * lastfixResult[i + 1];
        }
        return ans;
    }

    // 48. 旋转图像
    public void rotate(int[][] matrix) {
        int colLen = matrix.length;
        int rowLen = matrix.length;
        //上下交换
        for (int i = 0; i < rowLen / 2; i++) {
            for (int j = 0; j < colLen; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[colLen - 1 - i][j];
                matrix[colLen - 1 - i][j] = temp;
            }

        }

        // 主对角线交换
        for (int i = 1; i < colLen; i++) {
            int L = 0;
            int R = i;
            while (L < colLen && R < colLen) {

                int temp = matrix[L][R];
                matrix[L][R] = matrix[R][L];
                matrix[R][L] = temp;
                L++;
                R++;
            }
        }

    }

    //49. 字母异位词分组
    public List<List<String>> groupAnagrams2(String[] strs) {
        if (strs == null || strs.length == 0) return null;
        Map<String, List<String>> map = new HashMap<>();

        for (String str : strs) {
            char[] key = str.toCharArray();
            Arrays.sort(key);
            List<String> temp = map.getOrDefault(String.valueOf(key), new ArrayList<>());
            temp.add(str);
            map.put(String.valueOf(key), temp);
        }
        return new ArrayList<>(map.values());
    }


    // 56. 合并区间
    public int[][] merge(int[][] intervals) {
        int rowLen = intervals.length;
        Arrays.sort(intervals, (v1, v2) -> v1[0] - v2[0]);
        int[][] ans = new int[rowLen][2];
        int index = -1;


        for (int i = 0; i < rowLen; i++) {
            int L = intervals[i][0]; // 左边界
            int R = intervals[i][1];  //右边界
            if (index == -1 || L > ans[index][1]) ans[++index] = intervals[i];
            else ans[index][1] = Math.max(ans[index][1], R);

        }


        return Arrays.copyOf(ans, index + 1);

    }


    private void swap(int[] nums, int a, int b) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }

    //78. 子集 (返回一个数组所有子集)
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) return null;

        List<List<Integer>> ans = new ArrayList<>();

        subesetsBack(ans, 0, nums, new ArrayList<>());
        return ans;
    }

    private void subesetsBack(List<List<Integer>> ans, int index, int[] nums, List<Integer> temp) {
        if (index == nums.length) {
            ans.add(new ArrayList<>(temp));
            return;
        }
        //不选择该元素
        subesetsBack(ans, index + 1, nums, temp);

        //选择该元素

        temp.add(nums[index]);
        subesetsBack(ans, index + 1, nums, temp);
        temp.remove(temp.size() - 1);
    }

    //78. 子集 (返回一个数组所有子集) 方法二 : 非递归
    public List<List<Integer>> subsetsWay2(int[] nums) {
        if (nums == null || nums.length == 0) return null;

        List<List<Integer>> ans = new ArrayList<>();
        ans.add(new ArrayList<>());

        for (int num : nums) {
            int size = ans.size();
            for (int i = 0; i < size; i++) {
                //将ans中每一个集合末尾加入num形成新的集合
                List<Integer> integers = ans.get(i);
                ArrayList<Integer> newList = new ArrayList<>(integers);
                newList.add(num);
                //收集这些新的集合
                ans.add(newList);
            }
        }
        return ans;
    }

    // 79. 单词搜索

    boolean visit[][];

    public boolean exist(char[][] board, String word) {
        if (word.length() == 0 || board.length == 0) return false;
        visit = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                boolean flag = fun(board, i, j, 0, word);
                if (flag) return true;
            }
        }
        return false;
    }

    private boolean fun(char[][] board, int i, int j, int index, String word) {
        if (index == word.length()) {
            return true;
        }
        if (board[i][j] != word.charAt(index)) return false;


        visit[i][j] = true;
        //往上走
        if (i - 1 >= 0 && visit[i - 1][j] == false) {

            boolean flag = fun(board, i - 1, j, index + 1, word);
            if (flag) return true;

        }

        //往左走
        if (j - 1 >= 0 && visit[i][j - 1] == false) {

            boolean flag = fun(board, i, j - 1, index + 1, word);
            if (flag) return true;

        }
        //往下走
        if (i + 1 < board.length && visit[i + 1][j] == false) {

            boolean flag = fun(board, i + 1, j, index + 1, word);
            if (flag) return true;

        }
        if (j + 1 < board[0].length && visit[i][j + 1] == false) {
            //往右走

            boolean flag = fun(board, i, j + 1, index + 1, word);
            if (flag) return true;
        }
        visit[i][j] = false;
        return false;

    }

    // 94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {

        List<Integer> ans = new ArrayList();
        inorder(root, ans);
        return ans;
    }

    private void inorder(TreeNode node, List<Integer> ans) {
        if (node != null) {
            inorder(node.left, ans);
            ans.add(node.val);
            inorder(node.left, ans);
        }
    }

    // 96. 不同的二叉搜索树
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    // 98. 验证二叉搜索树
    long preValue = Long.MIN_VALUE;
    public boolean flag = true;

    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        fun(root);
        return flag;


    }

    private void fun(TreeNode node) {

        if (node != null) {
            fun(node.left);

            if (node.val <= preValue) {
                flag = false;

            }
            preValue = node.val;
            fun(node.right);
        }
    }

    // 101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return fun(root.left, root.right);
    }

    private boolean fun(TreeNode L, TreeNode R) {


        if (L == null && R == null) return true;
        if (L == null || R == null || L.val != R.val) return false;

        return fun(L.left, R.right) && fun(L.right, R.left);

    }

    // 102. 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {

                TreeNode node = queue.poll();

                temp.add(node.val);

                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            ans.add(temp);
        }
        return ans;
    }

    // 104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return fun2(root);

    }

    private int fun2(TreeNode node) {
        if (node == null) return 0;
        int left = fun2(node.left);
        int right = fun2(node.right);
        return 1 + Math.max(left, right);
    }

    // 105. 从前序与中序遍历序列构造二叉树
    private Map<Integer, Integer> inorderMap;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        inorderMap = new HashMap();
        int N = inorder.length;
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }

        return fun(preorder, 0, N - 1, 0, N - 1);
    }

    private TreeNode fun(int[] preorder, int preLeft, int preRight, int inLeft, int inRight) {
        if (preLeft > preRight) return null;

        int rootval = preorder[preLeft];
        TreeNode root = new TreeNode(rootval);

        int rootIndex = inorderMap.get(preorder[preLeft]);
        int leftSum = rootIndex - inLeft;

        //构建左子树
        root.left = fun(preorder, preLeft + 1, preLeft + leftSum, inLeft, rootIndex - 1);
        //构建右子树
        root.right = fun(preorder, preLeft + leftSum + 1, preRight, rootIndex + 1, inRight);
        return root;
    }

    // 114.二叉树展开为链表
    public void flatten(TreeNode root) {
        if (root == null) return;

        TreeNode cur = root;
        while (cur != null) {
            if (cur.left == null) cur = cur.right;

            TreeNode left = cur.left;
            TreeNode rightmostNode = left;
            while (rightmostNode.right != null) {
                rightmostNode = rightmostNode.right;
            }


            rightmostNode.right = cur.right;

            cur.right = left;
            cur.left = null;
            cur = cur.right;
        }

    }


    // 124. 二叉树中的最大路径和
    int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        fun5(root);
        return maxSum;
    }

    class Info {
        public int gain;

        public Info(int a) {
            if (a > 0) gain = a;
            else gain = 0;
        }
    }

    public Info fun5(TreeNode node) {
        if (node == null) return new Info(0);

        Info left = fun5(node.left);
        Info right = fun5(node.right);
        //以这个节点为根
        int sum = node.val + left.gain + right.gain;
        //不需要
        maxSum = Math.max(sum, maxSum);
        //不以这个为根
        return new Info(node.val + Math.max(left.gain, right.gain));
    }

    // 128. 最长连续序列
    public int longestConsecutinve(int[] nums) {
        Set<Integer> num_set = new HashSet<>();
        for (int num : nums) {
            num_set.add(num);
        }
        int max = 1;
        for (int num : nums) {
            if (num_set.contains(num - 1)) continue;
            int temp = num;
            int len = 1;
            while (num_set.contains(temp + 1)) {
                temp++;
                len++;
            }
            max = Math.max(len, max);
        }
        return max;
    }


    // 139. 单词拆分  动态规划
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];

        Arrays.fill(dp, false);
        dp[0] = true;

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (wordDict.contains(s.substring(j, i)) && dp[j]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];


    }

    // 141. 环形链表 1
    public boolean hasCycle(ListNode head) {

        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }

    // 142. 环形链表 II
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    slow = slow.next;
                    fast = fast.next;
                }
                return fast;
            }

        }

// 无环
        return null;

    }

    // 148. 排序链表  分治算法(递归版本)
    public ListNode sortList(ListNode head) {
        return sortList(head, null);
    }

    private ListNode sortList(ListNode l, ListNode r) {
        if (l == null) return l;
        if (l.next == r) {
            l.next = null;
            return l;
        }
        ListNode slow = l;
        ListNode fast = l;
        while (fast != r) {
            fast = fast.next;
            slow = slow.next;
            if (fast != r) {
                fast = fast.next;
            }
        }

        ListNode mid = slow;
        ListNode list1 = sortList(l, mid);
        ListNode list2 = sortList(mid, r);
        ListNode sorted = merge(list1, list2);
        return sorted;

    }

    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if (temp1 != null) {
            temp.next = temp1;
        } else if (temp2 != null) {
            temp.next = temp2;
        }
        return dummyHead.next;

    }


    // 160. 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode pA = headA, pB = headB;
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;

    }


    // 198. 打家劫舍  空间复杂度O(n),时间复杂度O(n)
    public int rob(int[] nums) {
        int len = nums.length;
        //[0,i)中最大的金额
        int[] dp = new int[len + 1];
        dp[0] = 0;
        dp[1] = nums[0];

        for (int i = 1; i < len; i++) {
            dp[i + 1] = Math.max(dp[i - 1] + nums[i], dp[i]);
        }


        return dp[len];
    }


    //827 最大人工岛
    public int largestIsland(int[][] grid) {
        if (grid == null || grid.length == 0) return 1;
        HashMap<Integer, Integer> map = new HashMap<>();
        int rowLen = grid.length;
        int colLen = grid[0].length;
        int indexOfIsland = 2;
        int curAreaOfIsland = 0;
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < colLen; j++) {
                if (grid[i][j] == 1) {
                    curAreaOfIsland = dfs(grid, i, j, indexOfIsland);
                    map.put(indexOfIsland++, curAreaOfIsland);
                }
            }
        }
        if (map.size() == 0) return 1;
        int ans = 0;
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < colLen; j++) {
                if (grid[i][j] == 0) {
                    Set<Integer> islands = getIslands(grid, i, j);
                    if (islands.size() != 0) {
                        ans = Math.max(ans, islands.stream().map(item -> map.get(item)).reduce(Integer::sum).orElse(0) + 1);
                    }

                }

            }

        }
        if (ans == 0) return map.get(2);
        return ans;
    }

    public boolean isLegal(int[][] grid, int row, int column) {
        return row >= 0 && row < grid.length && column >= 0 && column < grid[0].length;
    }


    public Set<Integer> getIslands(int[][] grid, int row, int column) {
        Set<Integer> result = new HashSet<>();
        if (isLegal(grid, row + 1, column) && grid[row + 1][column] != 0) result.add(grid[row + 1][column]);
        if (isLegal(grid, row - 1, column) && grid[row - 1][column] != 0) result.add(grid[row - 1][column]);
        if (isLegal(grid, row, column - 1) && grid[row][column - 1] != 0) result.add(grid[row][column - 1]);
        if (isLegal(grid, row, column + 1) && grid[row][column + 1] != 0) result.add(grid[row][column + 1]);
        return result;
    }


    private int dfs(int[][] grid, int i, int j, int indexOfIsland) {
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        if (i < 0 || j < 0 || i >= rowLen || j >= cowLen) return 0;
        if (grid[i][j] == indexOfIsland) return 0;
        if (grid[i][j] == 0) return 0;
        grid[i][j] = indexOfIsland;
        int ans = 1;
        //向四个方向扩散
        ans += dfs(grid, i - 1, j, indexOfIsland);
        ans += dfs(grid, i + 1, j, indexOfIsland);
        ans += dfs(grid, i, j - 1, indexOfIsland);
        ans += dfs(grid, i, j + 1, indexOfIsland);
        return ans;
    }


    //105.最大面积
    public int maxAreaOfIsland(int[][] grid) {
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        int answer = 0;
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < cowLen; j++) {

                answer = Math.max(answer, dfs(grid, i, j));
            }
        }
        return answer;
    }

    private int dfs(int[][] grid, int i, int j) {
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        if (i < 0 || j < 0 || i >= rowLen || j >= cowLen) return 0;
        if (grid[i][j] == 2) return 0;
        if (grid[i][j] == 0) return 0;
        grid[i][j] = 2;
        int ans = 1;
        //向四个方向扩散
        ans += dfs(grid, i - 1, j);
        ans += dfs(grid, i + 1, j);
        ans += dfs(grid, i, j - 1);
        ans += dfs(grid, i, j + 1);
        return ans;
    }




    //206. 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode last = null;
        while (cur != null) {
            ListNode nextNode = cur.next;
            cur.next = last;
            last = cur;
            cur = nextNode;
        }
        return last;
    }

    //207. 课程表
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> edges = new ArrayList<>();
        for (int i = 0; i < numCourses; i++)
            edges.add(new ArrayList<>());
        int[] visited = new int[numCourses];
        for (int[] cp : prerequisites)
            edges.get(cp[1]).add(cp[0]);
        for (int i = 0; i < numCourses; i++) {
            System.out.println(Arrays.toString(edges.get(i).toArray()));
        }
        for (int i = 0; i < numCourses; i++)
            if (dfs(edges, visited, i) == false) return false;
        return true;

    }

    private boolean dfs(List<List<Integer>> edges, int[] visited, int i) {
        if (visited[i] == 1) return false; //有环
        if (visited[i] == -1) return true; //没有换,已被遍历
        visited[i] = 1; //被访问过
        for (Integer j : edges.get(i))
            if (!dfs(edges, visited, j)) return false;
        visited[i] = -1;
        return true;
    }

    // 208 TRIE树
    class Trie {
        private Trie[] next;
        private boolean isEnd;

        public Trie() {
            next = new Trie[26];
            isEnd = false;
        }

        public void insert(String word) {
            Trie node = this;
            for (int i = 0; i < word.length(); i++) {
                char ch = word.charAt(i);
                int index = ch - 'a';
                if (node.next[index] == null) {
                    node.next[index] = new Trie();
                }
                node = node.next[index];
            }
            node.isEnd = true;
        }


        private boolean searchWith(String prefix) {
            Trie node = this;
            for (int i = 0; i < prefix.length(); i++) {
                char ch = prefix.charAt(i);
                int index = ch - 'a';
                if (node.next[index] == null) {
                    return false;
                }
                node = node.next[index];
            }
            return true;
        }


    }

    //215. 数组中的第K个最大元素 (优先队列)
    public int findKthLargest(int[] nums, int k) {
//       Collections.reverseOrder() 逆序排序比较器
        Queue<Integer> minQueue = new PriorityQueue<>(k, (v1, v2) -> (v1 - v2));

        for (int num : nums) {
            if (num < minQueue.peek()) continue;
            minQueue.add(num);

            if (minQueue.size() > k) {
                minQueue.poll();
            }
        }
        return minQueue.peek();


    }

    //215. 数组中的第K个最大元素 (快排思想) 时间复杂度O(n)
    public int findKthLargest2(int[] nums, int k) {
        return quickSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    public int quickSelect(int[] a, int l, int r, int index) {
        int q = partition(a, l, r);
        if (q == index) {
            return a[q];
        } else {
            return q < index ? quickSelect(a, q + 1, r, index) : quickSelect(a, l, q - 1, index);
        }
    }


    public int partition(int[] nums, int L, int R) {
        int j = L;
        int randomIndex = L + new Random().nextInt(R - L + 1);
        swap(nums, L, randomIndex);

        int pivot = nums[L];
        for (int i = L + 1; i <= R; i++) {
            if (nums[i] < pivot) {
                j++;
                swap(nums, i, j);
            }

        }
        swap(nums, L, j);
        return j;
    }

    //221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        int ans = 0;
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return ans;
        }
        int rowLen = matrix.length, colLen = matrix[0].length;
        int[][] dp = new int[rowLen][colLen];
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < colLen; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    ans = Math.max(ans, dp[i][j]);
                }
            }
        }
        int maxSquare = ans * ans;
        return maxSquare;


    }

    // 226. 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    //234. 回文链表 时间O(N) ,空间O(1)
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        //从slow开始反转链表
        ListNode cur = slow;
        ListNode pre = null;
        ListNode next = null;
        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        slow = head;
        while (pre != null && slow != null) {
            if (pre.val != slow.val) return false;
            pre = pre.next;
            slow = slow.next;
        }
        return true;


    }

    //236. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;

    }

    // 162.寻找局部最大值
    public int findPeakElement(int[] nums) {
        if (nums == null || nums.length == 0) return -1;
        int len = nums.length;
        if (len == 1 || nums[0] > nums[1]) return 0;
        if (nums[len - 1] > nums[len - 2]) return len - 1;
        int L = 1;
        int R = len - 2;
        int mid = 0;
        while (L < R) {
            mid = (L + R) / 2;
            if (nums[mid] < nums[mid - 1]) R = mid - 1;
            else if (nums[mid] < nums[mid + 1]) L = mid + 1;
            else return mid;
        }
        return L;
    }

    // 剑指 Offer 18. 删除链表的节点 (可以删除所有值=val的节点)
    public ListNode deleteNode(ListNode head, int val) {
        while (head != null) {
            if (head.val != val) break;
            head = head.next;
        }

        ListNode pre = head;
        ListNode cur = head;
        while (cur != null) {
            if (cur.val == val) {
                pre.next = cur.next;

            } else {
                pre = cur;
            }
            cur = cur.next;
        }
        return head;

    }

    //225. 用两个队列实现栈
    class MyStack {
        Queue<Integer> q1;
        Queue<Integer> q2;

        public MyStack() {
            q1 = new LinkedList<>();
            q2 = new LinkedList<>();
        }

        public void push(int x) {
            q2.add(x);
            while (!q1.isEmpty()) {
                q2.add(q1.poll());
            }
            Queue<Integer> temp = q1;
            q1 = q2;
            q2 = temp;
        }

        public int pop() {
            return q1.poll();
        }

        public int top() {
            return q1.peek();
        }

        public boolean empty() {
            return q1.isEmpty();
        }
    }



    //225. 用一个队列实现栈
    class MyStack2 {
        Queue<Integer> queue;

        /**
         * Initialize your data structure here.
         */
        public MyStack2() {
            queue = new LinkedList<Integer>();
        }

        /**
         * Push element x onto stack.
         */
        public void push(int x) {
            int n = queue.size();
            queue.offer(x);
            for (int i = 0; i < n; i++) {
                queue.offer(queue.poll());
            }
        }

        /**
         * Removes the element on top of the stack and returns that element.
         */
        public int pop() {
            return queue.poll();
        }

        /**
         * Get the top element.
         */
        public int top() {
            return queue.peek();
        }

        /**
         * Returns whether the stack is empty.
         */
        public boolean empty() {
            return queue.isEmpty();
        }
    }

    // 239. 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length < 2) return nums;

        LinkedList<Integer> q = new LinkedList<>();
        int len = nums.length;
        int[] ans = new int[nums.length - k + 1];

        for (int i = 0; i < len; i++) {
            while (!q.isEmpty() && nums[q.peekLast()] <= nums[i]) {
                q.pollLast();

            }
            q.addLast(i);
            if (q.peek() < i - k + 1) {
                q.poll();
            }
            if (i + 1 >= k) {
                ans[i + 1 - k] = nums[q.peek()];
            }
        }
        return ans;
    }

    // 剑指 Offer 09. 用两个栈实现队列
    class CQueue {

        Stack<Integer> stack_primary, stack2;

        public CQueue() {
            stack_primary = new Stack();
            stack2 = new Stack();
        }

        public void appendTail(int value) {
            stack_primary.add(value);
        }

        public int deleteHead() {
            if (stack2.isEmpty()) {
                while (!stack_primary.isEmpty()) stack2.push(stack_primary.pop());
            }

            if (stack2.isEmpty()) return -1;
            else return stack2.pop();
        }
    }

    // 240. 搜索二维矩阵 II
    public boolean searchMatrix(int[][] matrix, int target) {
        int rowlen = matrix.length;
        int collen = matrix[0].length;
        int i = 0;
        int j = collen - 1;
        while (i < rowlen && j >= 0) {
            if (target < matrix[i][j]) {
                j--;
            } else if (target > matrix[i][j]) {
                i++;
            } else return true;
        }
        return false;
    }


    //283. 移动零
    public void moveZeroes(int[] nums) {
        int len = nums.length;
        int j = 0;
//        [0,j]都是不等于0的
        for (int i = 0; i < len; i++) {
            if (nums[i] != 0) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
                j++;
            }

        }
    }


    // 662. 二叉树最大宽度 (广度优先遍历)
    public int widthOfBinaryTree(TreeNode root) {
        if (root == null) return 0;
        int ans = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(new TreeNode(1, root.left, root.right));


        while (!queue.isEmpty()) {
            int size = queue.size(), startIndex = -1, endIndex = -1;
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                endIndex = node.val;
                if (startIndex == -1) startIndex = node.val;

                if (node.left != null) {
                    TreeNode leftChildrenNode = node.left;
                    queue.add(new TreeNode(node.val * 2, leftChildrenNode.left, leftChildrenNode.right));
                }
                if (node.right != null) {
                    TreeNode rightChildrenNode = node.right;
                    queue.add(new TreeNode(node.val * 2, rightChildrenNode.left, rightChildrenNode.right));
                }
                ans = Math.max(ans, endIndex - startIndex + 1);
            }
        }
        return ans;
    }

    // 662. 二叉树最大宽度 (深度度优先遍历)
    int ans = 0;
    Map<Integer, Integer> minValue = new HashMap<>();

    public int widthOfBinaryTree2(TreeNode root) {
        dfs(root, 1, 0);
        return ans;
    }

    private void dfs(TreeNode node, int nodeIndex, int level) {
        if (node == null) return;
        minValue.putIfAbsent(level, nodeIndex);
        ans = Math.max(ans, nodeIndex - minValue.get(level) + 1);

        dfs(node.left, nodeIndex << 1, level + 1);
        dfs(node.right, nodeIndex << 1 | 1, level + 1);
    }

    //297. 二叉树的序列化与反序列化
    class Codec {


        public String serialize(TreeNode root) {
            return rSerialize(root);

        }

        private String rSerialize(TreeNode node) {
            if (node == null) {
                return "null,";
            }
            String str = node.val + ",";
            str += rSerialize(node.left);
            str += rSerialize(node.right);

            return str;
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            String[] value = data.split(",");
            Queue<String> queue = new LinkedList<>(Arrays.asList(value));
            return reconPreOrder(queue);

        }

        private TreeNode reconPreOrder(Queue<String> q) {
            if (q.isEmpty()) return null;
            String value = q.poll();
            if (value.equals("null")) {
                return null;
            }
            TreeNode head = new TreeNode(Integer.valueOf(value));
            head.left = reconPreOrder(q);
            head.right = reconPreOrder(q);
            return head;
        }
    }


    // 301. 删除无效的括号
    List<String> res = new ArrayList<>();

    public List<String> removeInvalidParentheses(String s) {
        int Lremove = 0;
        int Rremove = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                Lremove++;
            } else if (s.charAt(i) == ')') {
                if (Lremove == 0) Rremove++;
                else Lremove--;
            }
        }
        helper(s, 0, Lremove, Rremove);
        return res;


    }

    private void helper(String str, int start, int Lremove, int Rremove) {
        if (Lremove == 0 && Rremove == 0) {
            if (isValid(str)) {
                res.add(str);
            }
            return;
        }
        for (int i = start; i < str.length(); i++) {
            if (i != start && str.charAt(i) == str.charAt(i - 1)) {
                continue;
            }
            if (Lremove + Rremove > str.length() - i) {
                return;
            }
            if (Lremove > 0 && str.charAt(i) == '(') {
                helper(str.substring(0, i) + str.substring(i + 1), i, Lremove - 1, Rremove);
            }
            if (Rremove > 0 && str.charAt(i) == ')') {
                helper(str.substring(0, i) + str.substring(i + 1), i, Lremove, Rremove - 1);
            }

        }
    }
    public boolean isValid(String s) {
        char[] arr = s.toCharArray();
        Stack<Character> stack = new  Stack<>();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '(') stack.push(')');
            else if (arr[i] == '[') stack.push(']');
            else if (arr[i] == '{') stack.push('}');
            else {
                if (stack.isEmpty() || stack.peek() != arr[i]) {
                    return false;
                }
                stack.pop();
            }
        }
        return stack.isEmpty();
    }
    // 309. 最佳买卖股票时机含冷冻期
    public int maxProfit2(int[] prices) {
        int len = prices.length;

        int[][] dp = new int[len][3]; //dp[i][0] 表示持有股票 的最大收益 dp[i][1]表示

        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        dp[0][2] = 0;
        for (int i = 1; i < len; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
            dp[i][1] = dp[i - 1][0] + prices[i];
            dp[i][2] = Math.max(dp[i - 1][1], dp[i - 1][2]);
        }
        return Math.max(dp[len - 1][1], dp[len - 1][2]);
    }

    // 310. 最小高度树   (背题)
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            return Collections.singletonList(0);
        }

        List<Set<Integer>> adj = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            adj.add(new HashSet<>());
        }

        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }

        List<Integer> leaves = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (adj.get(i).size() == 1) {
                leaves.add(i);
            }
        }

        while (n > 2) {
            n -= leaves.size();
            List<Integer> newLeaves = new ArrayList<>();
            for (int i : leaves) {
                int j = adj.get(i).iterator().next();
                adj.get(j).remove(i);
                if (adj.get(j).size() == 1) {
                    newLeaves.add(j);
                }
            }
            leaves = newLeaves;
        }

        return leaves;
    }

    //312. 戳气球(背题)
    public int maxCoins(int[] nums) {
        int len = nums.length;
        int[][] dp = new int[len + 2][2 + len];
        int[] val = new int[2 + len];
        val[0] = val[1 + len] = 1;

        for (int i = 1; i <= len; i++) {
            val[i] = nums[i - 1];
        }
        for (int i = len - 1; i >= 0; i--) { // 最后一个,到 0
            for (int j = i + 2; j <= 1 + len; j++) { // i+2 -> 1+len
                for (int k = i + 1; k < j; k++) { // i+1 -> j)
                    int sum = val[i] * val[k] * val[j];
                    sum += dp[i][k] + dp[k][j];
                    dp[i][j] = Math.max(dp[i][j], sum);
                }
            }
        }
        return dp[0][len + 1];
    }


    // 338. 比特位计数
    public int[] countBits(int n) {
        int[] ans = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            ans[i] = ans[i >> 1] + (i & 1);
        }
        return ans;
    }

    // 337. 打家劫舍 III
    public int rob(TreeNode root) {
        Info2 ans = fun3(root);
        return Math.max(ans.noMax, ans.yesMax);
    }

    class Info2 {
        int noMax;

        int yesMax;

        public Info2(int noMax, int yesMax) {
            this.noMax = noMax > 0 ? noMax : 0;
            this.yesMax = yesMax > 0 ? yesMax : 0;
        }

    }

    private Info2 fun3(TreeNode node) {
        if (node == null) return new Info2(0, 0);

        Info2 leftInfo = fun3(node.left);
        Info2 rightInfo = fun3(node.right);

        //1.不要该节点:
        int temp = Math.max(leftInfo.yesMax, leftInfo.noMax);
        int temp2 = Math.max(rightInfo.yesMax, rightInfo.noMax);

        int noMax = temp2 + temp;
        //2.要该节点
        int yesMax = node.val + leftInfo.noMax + rightInfo.noMax;
        return new Info2(noMax, yesMax);
    }


    // 399. 除法求值
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int equationsSize = equations.size();

        UnionFind_399 unionFind = new UnionFind_399(2 * equationsSize);
        // 第 1 步：预处理，将变量的值与 id 进行映射，使得并查集的底层使用数组实现，方便编码
        Map<String, Integer> hashMap = new HashMap<>(2 * equationsSize);
        int id = 0;
        for (int i = 0; i < equationsSize; i++) {
            List<String> equation = equations.get(i);
            String var1 = equation.get(0);
            String var2 = equation.get(1);

            if (!hashMap.containsKey(var1)) {
                hashMap.put(var1, id);
                id++;
            }
            if (!hashMap.containsKey(var2)) {
                hashMap.put(var2, id);
                id++;
            }
            unionFind.union(hashMap.get(var1), hashMap.get(var2), values[i]);
        }

        // 第 2 步：做查询
        int queriesSize = queries.size();
        double[] res = new double[queriesSize];
        for (int i = 0; i < queriesSize; i++) {
            String var1 = queries.get(i).get(0);
            String var2 = queries.get(i).get(1);

            Integer id1 = hashMap.get(var1);
            Integer id2 = hashMap.get(var2);

            if (id1 == null || id2 == null) {
                res[i] = -1.0d;
            } else {
                res[i] = unionFind.isConnected(id1, id2);
            }
        }
        return res;
    }

    private class UnionFind_399 {

        private int[] parent;

        /**
         * 指向的父结点的权值
         */
        private double[] weight;


        public UnionFind_399(int n) {
            this.parent = new int[n];
            this.weight = new double[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                weight[i] = 1.0d;
            }
        }

        public void union(int x, int y, double value) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return;
            }

            parent[rootX] = rootY;
            // 关系式的推导请见「参考代码」下方的示意图
            weight[rootX] = weight[y] * value / weight[x];
        }

        /**
         * 路径压缩
         *
         * @param x
         * @return 根结点的 id
         */
        public int find(int x) {
            if (x != parent[x]) {
                int origin = parent[x];
                parent[x] = find(parent[x]);
                weight[x] *= weight[origin];
            }
            return parent[x];
        }

        public double isConnected(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return weight[x] / weight[y];
            } else {
                return -1.0d;

            }
        }
    }

    // 406. 根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (int[] a, int[] b) -> {
            //按照身高从大到小排
            if (a[0] != b[0]) return b[0] - a[0];
                //身高相等的按照人数从小到大排
            else return a[1] - b[1];
        });

        List<int[]> ans = new ArrayList<>();
        //将每个人插入到其对应的位置
        for (int[] person : people) {
            ans.add(person[1], person);
        }

        return ans.toArray(new int[ans.size()][]);
    }


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
            if (fun2(nums, k, target, temp + nums[i], index - 1)) return true;
            used[i] = false;
            while (i > 0 && nums[i] == nums[i - 1]) i--;
        }
        return false;

    }


    //438 滑动窗口
    public List<Integer> findAnagrams(String s, String p) {
        char[] s_chars = s.toCharArray();
        char[] p_chars = p.toCharArray();
        List<Integer> ans = new ArrayList<>();

        if (s_chars.length < p_chars.length) return ans;

        int[] s_count = new int[26];
        int[] p_count = new int[26];

        for (int i = 0; i < p_chars.length; i++) {
            s_count[s_chars[i] - 'a']++;
            p_count[p_chars[i] - 'a']++;
        }

        if (Arrays.equals(s_count, p_count)) {
            ans.add(0);
        }
        for (int i = 0; i < s_chars.length - p_chars.length; i++) {
            s_count[s_chars[i] - 'a']--;
            s_count[s_chars[i + p_chars.length] - 'a']++;
            if (Arrays.equals(s_count, p_count)) {
                ans.add(i + 1);
            }
        }
        return ans;
    }

    //26
    public void reorderList(ListNode head) {
        if (head == null) return;
        ListNode midNode = middleNode(head);
        ListNode l1 = head;
        ListNode l2 = midNode.next;
        midNode.next = null;
        l2 = reverseList(l2);

        mergeList(l1, l2);

    }

    private ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    private void mergeList(ListNode node1, ListNode node2) {
        ListNode l1_temp = node1;
        ListNode l2_temp = node2;
        while (l1_temp != null && l2_temp != null) {
            l1_temp = node1.next;
            l2_temp = node2.next;
            node1.next = node2;
            node1 = l1_temp;

            node2.next = node1;
            node2 = l2_temp;
        }
    }

    //328
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode evenHead = head.next; //偶数结点
        ListNode oddHead = head;
        ListNode oddCurNode = head;
        ListNode evenCurNode = head;
        while (evenCurNode != null && evenCurNode.next != null) {

            oddCurNode.next = evenCurNode.next;
            oddCurNode = oddCurNode.next;
            evenCurNode.next = oddCurNode.next;


            evenCurNode = evenCurNode.next;


        }
        oddCurNode.next = evenHead;
        return oddHead;

    }

    //459
    public boolean repeatedSubstringPattern(String s) {
        int n = s.length();
        for (int i = 1; i * 2 <= n; ++i) {
            if (n % i == 0) {
                boolean match = true;
                for (int j = i; j < n; ++j) {
                    if (s.charAt(j) != s.charAt(j - i)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    return true;
                }
            }
        }
        return false;
    }

}

