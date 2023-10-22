package leetcodes2;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;

/**
 * @author 张壮
 * @description 华为手撕题目汇总
 * @since 2023/10/13 14:17
 **/
public class Solution {
    public static void main(String[] args) {
        Solution solution = new Solution();
        /*boolean ans = solution.patternMatching("bbb","xxxxxx");
        System.out.println(ans);*/
        String multiply = solution.multiply("123", "9898");

    }


    //
    public String multiply(String num1, String num2) {
        int[] tempResult = new int[num1.length() + num2.length()];

        // 相乘
        for (int i = num1.length()-1; i >=0 ; i--) {
            System.out.println( "第"+i+"轮：===");
            for (int j = num2.length()-1; j >=0 ; j--) {
                tempResult[i+j+1] += (num1.charAt(i) - '0') * (num2.charAt(j)-'0');
                System.out.println(Arrays.toString(tempResult));
            }
        }
        System.out.println(Arrays.toString(tempResult));
        // 处理进位问题,从个位向前进位,最前面一位不可能有进位
        for (int i = tempResult.length-1; i>0 ; i--) {
            // 大于9，需要进位
            if (tempResult[i] > 9){
                tempResult[i-1] += tempResult[i] / 10;
                tempResult[i] %= 10;
            }
        }
        System.out.println(Arrays.toString(tempResult));

        // 处理第一个0
        StringBuilder stringResult = new StringBuilder();
        // flag表示还未找到第一个非0
        boolean flag = false;
        for (int i = 0; i < tempResult.length; i++) {
            if (tempResult[i] == 0 && !flag){
                continue;
            } else {
                stringResult.append(tempResult[i]);
                flag = true;
            }
        }
        return stringResult.toString();

    }
    //面试题16.18 模式匹配
    public boolean patternMatching(String pattern, String value) {
        //1.给定字符串长度为0
        if (value.length() == 0) {
            //1.1模式串长度不为0
            if (pattern.length() != 0) {
                for (int i = 1; i < pattern.length(); i++) {
                    if (pattern.charAt(i) != pattern.charAt(i - 1)) return false;
                }
            }
            //1.2模式串长度为0或满足1.1
            return true;
        }

        if (pattern.length() == 0) return false;
        if (pattern.length() == 1) return true;
        char a = pattern.charAt(0);
        int sum_a = 0, sum_b = 0;
        for (int i = 0; i < pattern.length(); i++) {
            if (pattern.charAt(i) == a) sum_a++;
            else sum_b++;
        }
        int len_value = value.length();
        int len_pattern = pattern.length();
        //2.如果模式串中 b 的个数为0
        if (sum_b == 0) {
            if (len_value % sum_a != 0) return false;
            int len_real_a = len_value / sum_a;
            String first_real_a = value.substring(0, len_real_a);
            for (int i = first_real_a.length(); i < value.length(); i += first_real_a.length()) {
                if (!value.substring(i, i + first_real_a.length()).equals(first_real_a)) return false;
            }
            return true;
        }
        //3.模式串中a或b的个数都不为0,给定字符串也不为0
        for (int len_real_a = 0; len_real_a * sum_a <= value.length(); len_real_a++) {
            System.out.print("当前字符串a的长度为："+ len_real_a);
            int rest = value.length() - len_real_a * sum_a;
            if (rest % sum_b != 0) {
                System.out.println("字符串b的长度为："+(double)rest/sum_b+",不为整数，所以跳过该循环！");
                continue;
            }
            int len_real_b = rest / sum_b;
            System.out.println("，字符串b的长度为："+ len_real_b);

            int next_string_index = 0;
            boolean flag = true;
            String real_a = "";
            String real_b = "";
            for (char cur_char : pattern.toCharArray()) {
                if (cur_char == a) {
                    String cur_a = value.substring(next_string_index, next_string_index += len_real_a);
                    if (real_a.length() == 0) {
                        real_a = cur_a;
                        System.out.print("此时的字符串a为:【"+real_a+"】 ");
                    } else if (!real_a.equals(cur_a)) {
                        flag = false;
                        break;
                    }
                } else {
                    String cur_b = value.substring(next_string_index, next_string_index += len_real_b);
                    if (real_b.length() == 0) {
                        real_b = cur_b;
                        System.out.print("此时的字符串b为:【"+real_b+"】 ");
                    } else if (!real_b.equals(cur_b)) {
                        flag = false;
                        break;
                    }
                }

            }
            System.out.println();
            if (flag && !real_a.equals(real_b)) return true;
        }
        return false;

    }
    //739.每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        LinkedList<Integer> stack = new LinkedList<>(); //栈顶元素值最小
        stack.push(0);
        int[] ans= new int[temperatures.length];
        for(int i=1 ;i < temperatures .length;i++){
            while(stack.isEmpty()==false && temperatures[i] > temperatures[stack.peek()]){
                ans[stack.peek()]=i-stack.peek();
                stack.pop();
            }
            stack.push(i);
        }
        return ans;
    }
    //3.不含有重复字符的最长子串的长度
    public int lengthOfLongestSubstring(String s) {
        int ans = 0 ;
        int l=0,r=0;
        HashMap<Character,Integer> map = new HashMap();
        for(int i =0 ; i < s.length() ; i++){
            r= i;
            Character c =  s.charAt(i);
            if(map.containsKey(c)){
                l = Math.max(l,map.get(c)+1);
            }
            map.put(c,i);
            ans = Math.max(ans,r-l+1);
        }
        return ans;

    }
    //62.不同路径
    public int uniquePaths(int m, int n) {
        if(m <= 0 || n <= 0) return -1;
        int[][] dp = new int[m][n];


        for (int i = 0; i < n; i++)
            dp[0][i] = 1;
        for (int j = 0; j < m; j++)
            dp[j][0] = 1;

        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++) {
                dp[i][j] += (dp[i - 1][j] + dp[i][j - 1]);
            }
        return dp[m - 1][n - 1];
    }
}

