package written.examination.dewu_9_12;

import java.util.Arrays;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/12 10:09
 **/
public class Main {
  /*  public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int[] nums = new int[len];
        for (int i = 0; i < len; i++) {
            nums[i] = in.nextInt();
        }
        if (len < 2) return;

      *//*  int len = 5;
        int[] nums = new int[]{2,1,2,1,2};*//*
        int[][] dp = new int[nums.length][2];
        dp[0][0] = 0;
        dp[0][1] = nums[0];
        dp[1][0] = nums[0];
        dp[1][1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < len; i++) {
            dp[i][0] =Math.max(dp[i-2][1], Math.max(dp[i - 1][0], dp[i - 1][1]));
            dp[i][1] = Math.max(nums[i] + dp[i - 1][0], Math.max(nums[i] + dp[i - 2][1], nums[i] + dp[i - 2][0]));
        }


        int ans = Math.max(dp[len - 1][0], dp[len - 1][1]);
        System.out.println(ans);
    }*/

    public static void main(String[] args) {
       Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int plusScore = in.nextInt();
        int[] nums = new int[len];
        for (int i = 0; i < len; i++) {
            nums[i] = in.nextInt();
        }
        int sum = 0;
      /*  int len = 5;
        int plusScore = 5;
        int[] nums = new int[]{1, 2, 3, 4, -6};*/

        Arrays.sort(nums);
        int i = 0;
        for (i = len - 1; i >= 0; i -= 3) {
            if (i < 0 || i - 1 < 0 || i - 2 < 0) break;
            int cur = nums[i] + nums[i - 1] + nums[i - 2];
            if (cur + plusScore >= 0) {
                sum += (cur + plusScore);
            }

        }
//        System.out.println(sum);
        if (i == -1) System.out.println(sum);
        else if (i == 0) {
            if (nums[0] > 0) sum += nums[0];
            System.out.println(sum);
        } else {
            if (nums[0] > 0) sum += nums[0];
            if (nums[1] > 0) sum += nums[1];
            System.out.println(sum);
        }


    }
}
/*
//情况1 i = -1   i<0;
2 1 0
//情况2  i=0 ,i-1<0
3 2 1 0
//情况3 i=1 ，i-2<0
4 3 2 1 0*/
