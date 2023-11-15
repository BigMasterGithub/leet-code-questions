package leetcodes.top100;

import com.kitfox.svg.A;
import data.structure.TreeNode;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author 张壮
 * @description leetcode top100 https://leetcode.cn/studyplan/top-100-liked/
 * @since 2023/10/14 16:51
 **/
public class Solution {
    int[][] direct = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
    boolean[][] vis;

    public static void main(String[] args) {
        Solution solution = new Solution();
        solution.firstMissingPositive(new int[]{-1, 1, 2, 3});
//        solution.rotate(new int[]{1,2,3,4,5,6,7},3);
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
//        System.out.println(solution.canPartitionKSubsets(new int[]{1, 5, 2, 9, 5, 3, 8}, 3));


//        MedianFinder medianFinder = new MedianFinder();
//        medianFinder.addNum(1);
//        medianFinder.addNum(5);
//        medianFinder.addNum(7);
//        medianFinder.addNum(4);
//        medianFinder.addNum(6);
//        medianFinder.findMedian();
//        medianFinder.addNum(3);
//        medianFinder.addNum(13);
//
//        medianFinder.addNum(23);
//
//        medianFinder.addNum(2);
//        solution.partitionLabels("ababcbacadefegdehijhklij");
//        solution.jump(new int[]{2,3,1,1,4});
//        solution.maxProfit_121(new int[]{2, 3, 8, 4, 7, 6, 43, 9, 1});
//        solution.topKFrequent(new int[]{1, 1, 1, 2, 2, 3}, 2);
//        solution.decodeString("3[a2[c]]");
//        solution.orangesRotting(new int[][]{{0}});
//        solution.searchRange(new int[]{2, 2, 3}, 4);
//        solution.generateParenthesis(3);
//        solution.minWindow2("a", "a");
    }

    //哈希
    //1.两数之和
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[]{-1, -1};
    }

    //49.字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
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

    //双指针
    //283. 移动零
    public void moveZeroes(int[] nums) {
        int len = nums.length;
        int j = 0;
        //[0,j]都是不等于0的
        for (int i = 0; i < len; i++) {
            if (nums[i] != 0) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
                j++;
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

    //滑动窗口系列
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


    //子串系列


    //560.和为K的子数组
    public int subarraySum(int[] nums, int k) {
        HashMap<Integer, Integer> preSumMap = new HashMap();
        preSumMap.put(0, 1);
        int ans = 0;
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (preSumMap.containsKey(sum - k)) {
                ans += preSumMap.get(sum - k);
            }
            preSumMap.put(sum, preSumMap.getOrDefault(sum, 0) + 1);
        }
        return ans;
    }


    // 239. 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length < 2) return nums;
        //（1）必须为双端队列，队列中元素为单调递减，队头元素 》 队尾元素
        LinkedList<Integer> q = new LinkedList<>();
        int len = nums.length;
        int[] ans = new int[nums.length - k + 1];

        for (int i = 0; i < len; i++) {
            //保证队列中元素为单调递减，队头元素 》 队尾元素，将当前元素放到合适位置
            while (!q.isEmpty() && nums[q.peekLast()] <= nums[i]) {
                q.pollLast();

            }

            q.addLast(i);
            //（3）保证队列中元素的下标范围在[i-k+1,i]内，
            if (q.peekFirst() < i - k + 1) {
                q.pollFirst();
            }
            //（4）i的值为k-1时，窗口已经形成。
            if (i >= k - 1) {
                ans[i - k + 1] = nums[q.peekFirst()];
            }
        }
        return ans;
    }

    //76.最小覆盖子串
    public String minWindow2(String s, String t) {
        String ansString = "";
        //(1)123的原因：A-65,Z-90,a-97,z-122
        int[] t_table = new int[123];
        for (int i = 0; i < t.length(); i++) {
            t_table[t.charAt(i)]++;
        }

        int l = 0, r = 0;
        int count = t.length();
        int ans = Integer.MAX_VALUE;
        while (r < s.length()) {
            if (t_table[s.charAt(r)] >= 1) count--;
            t_table[s.charAt(r)]--;

            //此时统计窗口中包含t的子串长度
            if (count == 0) {
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
                count++;
            }
            r++;
        }
        return ansString;

    }

    //普通数组
    //53.最大子数组和
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            dp[i] = nums[i];
        }
        dp[0] = nums[0];
        int ans = dp[0];

        for (int i = 1; i < nums.length; i++) {

            dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
            ans = Math.max(ans, dp[i]);

        }
        return ans;
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

    //189.轮转数组
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        fun_189(nums, 0, nums.length - 1);
        System.out.println(Arrays.toString(nums));
        fun_189(nums, k, nums.length - 1);
        System.out.println(Arrays.toString(nums));

        fun_189(nums, 0, k - 1);
        System.out.println(Arrays.toString(nums));

    }

    private void fun_189(int nums[], int L, int R) {
        while (L < R) {
            int temp = nums[L];
            nums[L] = nums[R];
            nums[R] = temp;
            L++;
            R--;
        }

    }

    //41.缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            System.out.println("开始前：" + Arrays.toString(nums) + "i = " + i);
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;

                System.out.println(Arrays.toString(nums));
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    //矩阵

    //73.矩阵置零
    public void setZeroes(int[][] matrix) {
        int rowLen = matrix.length;
        int colLen = matrix[0].length;
        boolean row0 = false, col0 = false;
        for (int i = 0; i < colLen; i++) {
            if (matrix[0][i] == 0) {
                row0 = true;
            }

        }
        for (int i = 0; i < rowLen; i++) {
            if (matrix[i][0] == 0) {
                col0 = true;
            }

        }
        for (int i = 1; i < rowLen; i++) {
            for (int j = 1; j < colLen; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < rowLen; i++) {
            for (int j = 1; j < colLen; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (col0) {
            for (int i = 0; i < rowLen; i++) {
                matrix[i][0] = 0;
            }
        }
        if (row0) {
            for (int j = 0; j < colLen; j++) {
                matrix[0][j] = 0;
            }
        }


    }

    //54.螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ans = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return ans;

        int up = 0, down = matrix.length - 1;
        int left = 0, right = matrix[0].length - 1;

        while (true) {
            for (int i = left; i <= right; i++) { // 左->右
                ans.add(matrix[up][i]);
            }
            if (++up > down) break;
            for (int i = up; i <= down; i++) { // 上->下
                ans.add(matrix[i][right]);
            }
            if (--right < left) break;
            for (int i = right; i >= left; i--) { // 右->左
                ans.add(matrix[down][i]);
            }
            if (--down < up) break;
            for (int i = down; i >= up; i--) { // 下->上
                ans.add(matrix[i][left]);
            }
            if (++left > right) break;
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

    // 240. 搜索二维矩阵 II  z字形查找法
    public boolean searchMatrix3(int[][] matrix, int target) {
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


    //二叉树系列

    //94.二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        fun_94(root, ans);
        return ans;
    }


    private void fun_94(TreeNode node, List<Integer> ans) {
        if (node == null) return;
        fun_94(node.left, ans);
        ans.add(node.val);
        fun_94(node.right, ans);
    }

    //104.二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return left > right ? left + 1 : right + 1;
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

    // 101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return fun_101(root.left, root.right);
    }

    private boolean fun_101(TreeNode L, TreeNode R) {


        if (L == null && R == null) return true;
        if (L == null || R == null || L.val != R.val) return false;

        return fun_101(L.left, R.right) && fun_101(L.right, R.left);

    }

    //543.二叉树的直径
    int ans = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        fun(root);
        return ans;
    }

    private int fun(TreeNode node) {
        if (node == null) return 0;
        int L = fun(node.left);
        int R = fun(node.right);
        ans = Math.max(ans, L + R);
        return Math.max(L, R) + 1;
    }

    //108.将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        TreeNode root = fun(nums, 0, nums.length - 1);
        return root;
    }

    private TreeNode fun(int[] nums, int l, int r) {
        if (l > r) return null;
        int mid = (l + r) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = fun(nums, 0, mid - 1);
        node.right = fun(nums, mid + 1, r);
        return node;
    }

    //98. 验证二叉搜索树
    long preValue = Long.MIN_VALUE;
    public boolean flag = true;

    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        fun_98(root);
        return flag;


    }

    private void fun_98(TreeNode node) {

        if (node != null) {
            fun_98(node.left);

            if (node.val <= preValue) {
                flag = false;

            }
            preValue = node.val;
            fun_98(node.right);
        }
    }

    //230.二叉搜索树中第K小的元素 - 中序迭代

    public int kthSmallest(TreeNode root, int k) {
        LinkedList<TreeNode> stack = new LinkedList<TreeNode>();
        int curNum = 0;
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            while (cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            curNum++;
            if (curNum == k) {
                return cur.val;
            }
            cur = cur.right;
        }
        return -1;
    }

    int num = 0;

    //230.二叉搜索树中第K小的元素 - 中序递归
    public int kthSmallest2(TreeNode root, int k) {
        fun2(root, k);
        return ans;
    }

    private void fun2(TreeNode node, int k) {
        if (node == null) return;
        fun2(node.left, k);
        num++;
        if (num == k) {
            ans = node.val;
        }
        fun2(node.right, k);
    }

    //119.二叉树的右视图 - BFS
    public List<Integer> rightSideView(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> ans = new ArrayList<>();
        if (root == null) return ans;
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode poll = queue.poll();
                if (i == 0) {
                    ans.add(poll.val);
                }
                if (poll.right != null) {
                    queue.add(poll.right);
                }
                if (poll.left != null) {
                    queue.add(poll.left);
                }
            }


        }
        return ans;
    }

    //119.二叉树的右视图 - DFS
    List<Integer> ans2 = new ArrayList<>();

    public List<Integer> rightSideView2(TreeNode root) {
        fun_119(root, 0);
        return ans2;
    }

    private void fun_119(TreeNode node, int depth) {
        if (node == null) return;
        if (depth == ans2.size()) {
            ans2.add(node.val);
        }
        depth++;
        fun_119(node.right, depth);
        fun_119(node.left, depth);
    }

    // 114.二叉树展开为链表 - 前序遍历版
    public void flatten(TreeNode root) {
        if (root == null) return;

        List<TreeNode> list = new ArrayList();

        fun(root, list);

        for (int i = 0; i < list.size() - 1; i++) {
            TreeNode cur = list.get(i);
            cur.right = list.get(i + 1);
            cur.left = null;
        }


    }

    void fun(TreeNode node, List<TreeNode> list) {
        if (node != null) {
            list.add(node);
            fun(node.left, list);
            fun(node.right, list);
        }
    }

    // 114.二叉树展开为链表 - 后序遍历
    public void flatten2(TreeNode root) {
        if (root == null) {
            return;
        }
        //将根节点的左子树变成链表
        flatten2(root.left);
        //将根节点的右子树变成链表
        flatten2(root.right);
        TreeNode temp = root.right;
        //把树的右边换成左边的链表
        root.right = root.left;
        //记得要将左边置空
        root.left = null;
        //找到树的最右边的节点
        while (root.right != null) root = root.right;
        //把右边的链表接到刚才树的最右边的节点
        root.right = temp;
    }

    // 114.二叉树展开为链表 - 迭代思想
    public void flatten3(TreeNode root) {
        while (root != null) {
            TreeNode curNode = root.left;
            while (curNode != null && curNode.right != null) {
                curNode = curNode.right;
            }

            if (curNode != null) {
                curNode.right = root.right;
                root.right = root.left;
                root.left = null;

            }
            root = root.right;
        }
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


        int rootIndex = inorderMap.get(preorder[preLeft]);
        int leftSum = rootIndex - inLeft;
        TreeNode root = new TreeNode(rootval);
        //构建左子树
        root.left = fun(preorder, preLeft + 1, preLeft + leftSum, inLeft, rootIndex - 1);
        //构建右子树
        root.right = fun(preorder, preLeft + leftSum + 1, preRight, rootIndex + 1, inRight);
        return root;
    }

    //437.路径总和Ⅲ --dfs
    int ans3 = 0;

    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) return 0;
        fun(root, 0, targetSum);
        pathSum(root.left, targetSum);
        pathSum(root.right, targetSum);
        return ans3;
    }

    private void fun(TreeNode node, int temp, int targetSum) {
        if (node == null) return;
        temp += node.val;
        if (temp == targetSum) {
            ans3++;
        }
        fun(node.left, temp, targetSum);
        fun(node.right, temp, targetSum);

    }

    //236.二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q || root == null) return root;
        TreeNode left_result = lowestCommonAncestor(root.left, p, q);
        TreeNode right_result = lowestCommonAncestor(root.right, p, q);
        if (left_result == null && right_result == null) return null;
        else if (left_result == null && right_result != null) return right_result;
        else if (left_result != null && right_result == null)
            return left_result;

        else return root;

    }

    // 124. 二叉树中的最大路径和
    int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        fun_124(root);
        return maxSum;
    }


    public int fun_124(TreeNode node) {
        if (node == null) return 0;

        int left = Math.max(fun_124(node.left), 0);
        int right = Math.max(fun_124(node.right), 0);
        //以这个节点为根
        int sum = node.val + left + right;

        maxSum = Math.max(sum, maxSum);
        //不以这个为根
        return node.val + Math.max(left, right);
    }

    //图论

    //200.岛屿数量

    //dfs解法
    public int numIslands(char[][] grid) {
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        int answer = 0;
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < cowLen; j++) {
                if (grid[i][j] == '1') answer++;
                dfs(grid, i, j);
            }
        }
        return answer;
    }

    private void dfs(char[][] grid, int i, int j) {
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        if (i < 0 || j < 0 || i >= rowLen || j >= cowLen || grid[i][j] == '0') {
            return;
        }
        if (grid[i][j] == '2') return;


        grid[i][j] = '2';

        //向四个方向扩散
        dfs(grid, i - 1, j);
        dfs(grid, i + 1, j);
        dfs(grid, i, j - 1);
        dfs(grid, i, j + 1);

    }

    //994.腐烂的橘子
    public int orangesRotting(int[][] grid) {
        int rowLen = grid.length;
        int colLen = grid[0].length;
        vis = new boolean[rowLen][colLen];
        LinkedList<int[]> queue = new LinkedList<>();
        int ans = 0;
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < colLen; j++) {
                if (grid[i][j] == 2) {
                    queue.add(new int[]{i, j});
                }
            }
        }
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int f = 0; f < size; f++) {
                int[] poll = queue.poll();
                for (int i = 0; i < 4; i++) {
                    int newX = poll[0] + direct[i][0];
                    int newY = poll[1] + direct[i][1];
                    if (newX < 0 || newX == grid.length || newY < 0 || newY == grid[0].length || grid[newX][newY] == 2) {
                        continue;
                    }
                    grid[newX][newY] = 2;
                    queue.add(new int[]{newX, newY});

                }

            }
            ans++;

        }
        if (ans > 0) ans--;
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < colLen; j++) {
                if (grid[i][j] == 1) {
                    return -1;
                }
            }
        }
        return ans;

    }

    //207.课程表 bfs
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        LinkedList<Integer> queue = new LinkedList();
        boolean ans = false;
        List<List<Integer>> adjacencyList = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            adjacencyList.add(new ArrayList<>());

        }

        int[] indegree = new int[numCourses];
        for (int i = 0; i < prerequisites.length; i++) {

            int curCourseIndex = prerequisites[i][0];
            int prerequisiteCourseIndex = prerequisites[i][1];
            adjacencyList.get(prerequisiteCourseIndex).add(curCourseIndex);

            indegree[curCourseIndex]++;
        }

        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) queue.offer(i);

        }
        while (!queue.isEmpty()) {
            Integer curprerequisiteCourseIndex = queue.poll();
            numCourses--;
            int size = adjacencyList.get(curprerequisiteCourseIndex).size();
            for (int i = 0; i < size; i++) {
                int courseIndex = adjacencyList.get(curprerequisiteCourseIndex).get(i);
                indegree[courseIndex]--;
                if (indegree[courseIndex] == 0) queue.add(courseIndex);
            }


        }

        ans = numCourses == 0;
        return ans;
    }

    //207.课程表 dfs
    public boolean canFinish2(int numCourses, int[][] prerequisites) {
        List<List<Integer>> edges = new ArrayList<>();
        for (int i = 0; i < numCourses; i++)
            edges.add(new ArrayList<>());
        int[] visited = new int[numCourses];
        for (int[] cp : prerequisites)
            edges.get(cp[1]).add(cp[0]);

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

    //208.实现Trie
    class Trie {

        private class Node {
            public Node[] next;
            public boolean isWord;

            public Node(boolean isWord) {
                this.isWord = isWord;
                next = new Node[26];
            }

            public Node() {
                this(false);
            }

        }

        Node root;

        public Trie() {
            root = new Node();
        }

        public void insert(String word) {
            Node cur = root;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (cur.next[c - 'a'] == null) {
                    cur.next[c - 'a'] = new Node();
                }
                cur = cur.next[c - 'a'];
            }
            cur.isWord = true;
        }

        public boolean search(String word) {
            Node cur = root;


            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (cur.next[c - 'a'] == null) return false;
                cur = cur.next[c - 'a'];
            }
            return cur.isWord;


        }

        public boolean startsWith(String prefix) {
            Node cur = root;
            for (int i = 0; i < prefix.length(); i++) {
                char c = prefix.charAt(i);
                if (cur.next[c - 'a'] == null) return false;
                cur = cur.next[c - 'a'];
            }
            return true;
        }
    }


    //回溯系列

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

    //78. 子集 (返回一个数组所有子集)
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) return null;

        List<List<Integer>> ans = new ArrayList<>();

        fun(ans, 0, nums, new ArrayList<>());
        return ans;
    }

    private void fun(List<List<Integer>> ans, int index, int[] nums, List<Integer> temp) {
        if (index == nums.length) {
            ans.add(new ArrayList<>(temp));
            return;
        }
        //不选择该元素
        fun(ans, index + 1, nums, temp);

        //选择该元素

        temp.add(nums[index]);
        fun(ans, index + 1, nums, temp);
        temp.remove(temp.size() - 1);
    }

    //17.电话号码的字母组合
    String[] numString = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList();

        StringBuilder cur = new StringBuilder();
        if (digits == null || digits.length() == 0) return ans;

        fun_17(digits, 0, cur, ans);
        return ans;
    }

    private void fun_17(String digits, int num, StringBuilder cur, List<String> ans) {
        if (num == digits.length()) {
            ans.add(cur.toString());
            return;
        }
        String curStr = numString[digits.charAt(num) - '0'];
        for (int i = 0; i < curStr.length(); i++) {
            cur.append(curStr.charAt(i));
            fun_17(digits, num + 1, cur, ans);
            cur.deleteCharAt(cur.length() - 1);
        }


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
            ans.add(new ArrayList<>(temp));
            return;
        }
        fun(arr, rest, index + 1, temp, ans);
        if (arr[index] <= rest) {
            temp.add(arr[index]);
            fun(arr, rest - arr[index], index, temp, ans);
            temp.remove(temp.size() - 1);//回溯算法的精髓所在

        }
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
            System.out.println(cur.toString());
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

    //79.单词搜索
    boolean[][] visited;

    public boolean exist(char[][] board, String word) {
        int rowLen = board.length;
        int colLen = board[0].length;
        visited = new boolean[rowLen][colLen];
        for (int i = 0; i < rowLen; i++) {
            for (int j = 0; j < colLen; j++) {
                boolean ans = fun_79(i, j, board, word, 0);
                if (ans) return true;
            }

        }
        return false;

    }

    private boolean fun_79(int i, int j, char[][] board, String word, int curIndex) {
        if (board[i][j] != word.charAt(curIndex)) return false;
        else if (curIndex == word.length() - 1) {
            return true;
        }
        visited[i][j] = true;
        boolean ans = false;
        for (int k = 0; k < 4; k++) {
            int newX = direct[k][0] + i;
            int newY = direct[k][1] + j;
            if (newX >= 0 && newX < board.length
                    && newY >= 0 && newY < board[0].length) {
                if (visited[newX][newY] == false) {
                    boolean curAns = fun_79(newX, newY, board, word, curIndex + 1);
                    if (curAns) {
                        ans = true;
                        break;
                    }
                }
            }

        }
        visited[i][j] = false;
        return ans;
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

    //51.N皇后
    List<List<String>> ans_51 = new ArrayList<>();

    public List<List<String>> solveNQueens(int n) {
        char[][] chessboard = new char[n][n];
        for (char[] c : chessboard) {
            Arrays.fill(c, '.');
        }
        fun_51(0, n, chessboard);
        return ans_51;
    }

    private void fun_51(int curRowIndex, int n, char[][] chessboard) {
        if (curRowIndex == n) {
            List<String> result = new ArrayList<>();
            for (char[] c : chessboard) {
                result.add(String.copyValueOf(c));
            }
            ans_51.add(result);
            return;
        }
        int rowLen = n;
        for (int i = 0; i < rowLen; i++) {
            if (isValid_51(curRowIndex, i, n, chessboard)) {
                chessboard[curRowIndex][i] = 'Q';
                fun_51(curRowIndex + 1, n, chessboard);
                chessboard[curRowIndex][i] = '.';
            }
        }

    }

    private boolean isValid_51(int curRowIndex, int curColIndex, int n, char[][] chessboard) {
        //检查【列】
        for (int k = 0; k < n; k++) {
            if (chessboard[k][curColIndex] == 'Q') {
                return false;
            }
        }
        //检查【主对角线】
        for (int i = curRowIndex-1,j=curColIndex-1; i>=0&&j>=0;i--,j--) {
                if(chessboard[i][j]=='Q'){
                    return false;
                }
        }
        //检查【斜对角线】
        for (int i = curRowIndex-1,j=curColIndex+1; i>=0&&j<n;i--,j++) {
            if(chessboard[i][j]=='Q'){
                return false;
            }
        }
        return true;
    }


    //二分查找

    //35.搜索插入位置
    public int searchInsert(int[] nums, int target) {
        int L = 0;
        int R = nums.length - 1;
        while (L <= R) {
            int mid = (L + R) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] < target) {
                L = mid + 1;
            } else {
                R = mid - 1;
            }
        }
        return L;
    }


    //74.搜索二维矩阵
    public boolean searchMatrix(int[][] matrix, int target) {
        int L = 0, R = matrix[0].length - 1;
        while (L < matrix.length && R >= 0) {
            int mid = matrix[L][R];
            if (mid == target) {
                return true;
            } else if (mid < matrix[L][R]) {
                L++;
            } else {
                R--;
            }
        }
        return false;
    }

    //74.搜索二维矩阵 -  真正的二分
    public boolean searchMatrix2(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int low = 0, high = m * n - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            int x = matrix[mid / n][mid % n];
            if (x < target) {
                low = mid + 1;
            } else if (x > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }

    //34.在排序数组中查找元素的第一个和最后一个位置

    //二分法:将问题转化为 x>target-1 的最左的x的下标和y>target 的最左的y的下标,
    //如果存在x=target的话,那么[x,  y-1]就是target 的[起始.终止]位置.
    public int[] searchRange(int[] nums, int target) {
        if (nums.length == 0) return new int[]{-1, -1};
        int index1 = bs(nums, target - 1);
        int index2 = bs(nums, target);
        System.out.println(index1 + "," + index2);
        return index1 <= index2 - 1 && nums[index1] == target ? new int[]{index1, index2 - 1} : new int[]{-1, -1};

    }

    private int bs(int[] nums, int tar) {
        int L = 0;
        int R = nums.length - 1;
        int index = nums.length;
        while (L <= R) {
            int mid = (L + R) / 2;
            if (nums[mid] > tar) {
                index = mid;
                R = mid - 1;

            } else L = mid + 1;


        }

        return index;
    }

    //33.搜索旋转排序数组
    public int search(int[] nums, int target) {

        int N = nums.length;
        if (N == 0) return -1;
        if (N == 1) return nums[0] == target ? 0 : -1;
        int L = 0, R = N - 1;
        while (L <= R) {
            int mid = (L + R) / 2;
            if (nums[mid] == target) return mid;
            if (nums[L] <= nums[mid]) {//mid左边是有序的
                if (nums[L] <= target && target < nums[mid])
                    R = mid - 1;
                else L = mid + 1;

            } else {//mid右边是有序的
                if (nums[mid] < target && target <= nums[R])
                    L = mid + 1;
                else
                    R = mid - 1;
            }
        }
        return -1;


    }

    //栈

    // 20. 有效的括号
    public boolean isValid(String s) {
        char[] arr = s.toCharArray();
        Stack<Character> stack = new Stack<>();
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

    //155. 最小栈
    class MinStack {
        Stack<Integer> dataStack;
        Stack<Integer> minStack;

        public MinStack() {
            dataStack = new Stack();
            minStack = new Stack();
        }

        public void push(int val) {
            dataStack.push(val);
            if (minStack.isEmpty()) {
                minStack.push(val);
            } else minStack.push(Math.min(minStack.peek(), val));
        }

        public void pop() {
            dataStack.pop();
            minStack.pop();
        }

        public int top() {
            return dataStack.peek();
        }

        public int getMin() {
            return minStack.peek();
        }
    }

    //394. 字符串解码
    public String decodeString(String s) {

        StringBuilder ans = new StringBuilder();
        int num = 0;
        LinkedList<Integer> stack_num = new LinkedList<>();
        LinkedList<String> stack_res = new LinkedList<>();
        for (Character c : s.toCharArray()) {
            if (c == '[') {
                stack_num.push(num);
                stack_res.push(ans.toString());
                num = 0;
                ans = new StringBuilder();
                System.out.println(Arrays.toString(stack_num.toArray(new Integer[0])));
                System.out.println(Arrays.toString(stack_res.toArray(new String[0])));
            } else if (c == ']') {
                StringBuilder temp = new StringBuilder();
                int curNum = stack_num.pop();
                for (int i = 0; i < curNum; i++) {
                    temp.append(ans);
                }
                ans = new StringBuilder(stack_res.pop() + temp);

            } else if (c >= '0' && c <= '9') num = num * 10 + Integer.parseInt(c + "");
            else ans.append(c);

        }
        return ans.toString();


    }

    //739 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
//       1.栈顶最小，栈底最大
        LinkedList<Integer> stack = new LinkedList<>();//
        stack.push(0);
        int[] ans = new int[temperatures.length];
        for (int i = 1; i < temperatures.length; i++) {
            while (stack.isEmpty() == false && temperatures[i] > temperatures[stack.peek()]) {
                ans[stack.peek()] = i - stack.peek();
                stack.pop();
            }
            stack.push(i);
        }
        return ans;
    }

    //84.柱状图形的最大的矩形
    public int largestRectangleArea(int[] heights) {
        int ans = heights[0];
        LinkedList<Integer> stack = new LinkedList<>();
        int[] heights_new = new int[heights.length + 2];
        for (int i = 0; i < heights.length; i++) {
            heights_new[1 + i] = heights[i];

        }
        heights_new[0] = 0;
        heights_new[heights.length + 1] = 0;
        stack.push(0);
        for (int i = 1; i < heights_new.length; i++) {
            while (stack.isEmpty() == false && heights_new[i] < heights_new[stack.peek()]) {
                int curHeight = heights_new[stack.pop()];
                int weight = i - stack.peek() - 1;
                ans = Math.max(ans, curHeight * weight);

            }
            stack.push(i);
        }


        return ans;
    }


    //堆

    //215. 数组中的第K个最大元素 (优先队列)
    public int findKthLargest(int[] nums, int k) {
        //       Collections.reverseOrder() 逆序排序比较器
        Queue<Integer> minQueue = new PriorityQueue<>(k, (v1, v2) -> (v1 - v2));

        for (int num : nums) {

            minQueue.add(num);

            if (minQueue.size() > k) {
                minQueue.poll();
            }
        }
        return minQueue.peek();
    }

    // 347. 前 K 个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> num_count = new HashMap<Integer, Integer>();
        for (int num : nums) {
            num_count.put(num, num_count.getOrDefault(num, 0) + 1);
        }
        Queue<int[]> queue = new PriorityQueue<>((int[] a, int[] b) -> a[1] - b[1]);
        // 注意map的遍历方法,性能高
        for (Map.Entry<Integer, Integer> cur : num_count.entrySet()) {
            int num = cur.getKey();
            int count = cur.getValue();
            if (queue.size() == k) {
                if (queue.peek()[1] < count) {
                    queue.poll();
                    queue.offer(new int[]{num, count});
                }
            } else {
                queue.offer(new int[]{num, count});
            }

        }

        int[] ans = new int[k];
        for (int i = 0; i < k; i++) {
            ans[i] = queue.poll()[0];
        }
        return ans;
    }


    //295.数据流的中位数
    static class MedianFinder {
        PriorityQueue<Integer> queMax;
        PriorityQueue<Integer> queMin;

        public MedianFinder() {
            queMax = new PriorityQueue<Integer>((a, b) -> (b - a));
            queMin = new PriorityQueue<Integer>(Comparator.comparingInt(a -> a));
        }

        public void addNum(int num) {
            if (queMax.isEmpty() || num <= queMax.peek()) {
                queMax.offer(num);
                if (queMin.size() + 1 < queMax.size()) {
                    queMin.offer(queMax.poll());
                }
            } else {
                queMin.offer(num);
                if (queMin.size() > queMax.size()) {
                    queMax.offer(queMin.poll());
                }
            }
            System.out.println(" --- ");
            Integer[] minArray = queMax.toArray(new Integer[0]);

            System.out.println(Arrays.toString(minArray));
            Integer[] maxArray = queMin.toArray(new Integer[0]);

            System.out.println(Arrays.toString(maxArray));
        }

        public double findMedian() {
            if (queMax.size() > queMin.size()) {
                return queMax.peek();
            }
            return (queMax.peek() + queMin.peek()) / 2.0;
        }

    }


    //贪心算法

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
        for (int i = 0; i < dp.length; i++) {
            System.out.print(Arrays.toString(dp[i]) + " ");
        }
        return dp[prices.length - 1][1];
    }

    //55.跳跃游戏
    public boolean canJump(int[] nums) {
        int end = 0;
        int maxPosition = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxPosition = Math.max(i + nums[i], maxPosition);
            System.out.printf("%d = %d：能跳的最远位置为：%d\n", i, nums[i], maxPosition);
            System.out.println(end);
            if (i == end) {
                end = maxPosition;
            }
        }
        return end >= nums.length - 1;
    }

    //45.跳跃游戏2
    public int jump(int[] nums) {
        int end = 0;
        int ans = 0;
        int maxPosition = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            maxPosition = Math.max(i + nums[i], maxPosition);
            System.out.printf("%d = %d：能跳的最远位置为：%d\n", i, nums[i], maxPosition);
            System.out.println(end);
            if (i == end) {
                ans++;
                end = maxPosition;
            }
        }
        System.out.println(ans);
        return ans;
    }

    //763.划分字母区间
    public List<Integer> partitionLabels(String s) {
        int[] last = new int[26];
        int length = s.length();
        for (int i = 0; i < length; i++) {
            last[s.charAt(i) - 'a'] = i;
        }
        for (int i = 0; i < 26; i++) {
            System.out.printf(" %c,", i + 'a');
        }
        System.out.println();
        System.out.println(Arrays.toString(last));
        List<Integer> ans = new ArrayList<Integer>();
        int start = 0, end = 0;
        for (int i = 0; i < length; i++) {
            System.out.printf(" %c,", s.charAt(i));

            end = Math.max(end, last[s.charAt(i) - 'a']);
            if (i == end) {
                ans.add(end - start + 1);
                start = end + 1;
            }
        }
        System.out.println();
        System.out.println(ans);
        return ans;


    }


    //额外题目!!!!
    // 698. 分为k个相等的子集 方法一：桶选球,复杂度O（2^n）^k
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
            System.out.println("选择" + nums[i]);
            if (fun2(nums, k, target, temp + nums[i], index - 1)) {

                return true;
            }
            used[i] = false;
            while (i > 0 && nums[i] == nums[i - 1]) i--;
        }
        return false;

    }

    //动态规划


    //118.杨辉三角
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        for (int i = 0; i < numRows; ++i) {
            List<Integer> row = new ArrayList<Integer>();
            for (int j = 0; j <= i; ++j) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    row.add(ret.get(i - 1).get(j - 1) + ret.get(i - 1).get(j));
                }
            }
            ret.add(row);
        }
        return ret;
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
        //dp[i]表示长度为i的情况下,能否由wordDict拼出。
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
        System.out.println("初始数组：" + Arrays.toString(nums));
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

    // 32. 最长有效括号
    // 方法一 : 常规解法
    public int longestValidParentheses1(String s) {
        int leftSum = 0;
        int rightSum = 0;
        int maxLen = 0;
        char[] arr = s.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '(') leftSum++;
            else rightSum++;
            if (leftSum == rightSum) {
                maxLen = Math.max(maxLen, leftSum + rightSum);
            }
            if (rightSum > leftSum) {
                leftSum = 0;
                rightSum = 0;
            }
        }
        leftSum = rightSum = 0;
        for (int i = arr.length - 1; i >= 0; i--) {
            if (arr[i] == '(') leftSum++;
            else rightSum++;
            if (leftSum == rightSum) {
                maxLen = Math.max(maxLen, leftSum + rightSum);
            }
            if (rightSum < leftSum) {
                leftSum = 0;
                rightSum = 0;
            }
        }
        return maxLen;
    }

    // 32. 最长有效括号
    // 方法二：动态规划
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
                if (L >= 0 && chars[L] == '(') dp[R] = 2 + ((L - 1 >= 0) ? dp[L - 1] : 0) + dp[R - 1];

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
        int rowLen = grid.length;
        int colLen = grid[0].length;

        int[][] dp = new int[rowLen][colLen];
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
    /*leetcode中回文子串系列问题：5，125，647，131，516，234*/
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
                    if (R == L + 1 || R == L + 2) {
                        dp[L][R] = true;
                    } else {
                        dp[L][R] = dp[L + 1][R - 1];
                    }
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

    //287. 寻找重复数
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

}

