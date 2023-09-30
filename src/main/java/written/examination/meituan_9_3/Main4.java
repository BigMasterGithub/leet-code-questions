package written.examination.meituan_9_3;

import java.util.Arrays;
import java.util.Scanner;
import java.util.TreeSet;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/2 19:46
 **/
public class Main4 {
    static TreeSet<Integer> curSet = new TreeSet<>((o1, o2) -> {
        return o2 - o1;
    });
    static int ans = 0;
    static int mode = 7 + 1000_000_000;

    public static void main(String[] args) {


        Scanner in = new Scanner(System.in);

        int len = in.nextInt();
        int deleteNum = in.nextInt();
        int[] nums = new int[len];
        for (int i = 0; i < len; i++) {
            nums[i] = in.nextInt();
        }
//        System.out.println(Arrays.toString(nums));
        int target = len - deleteNum;

//        int[] nums = new int[]{1, 2, 3, 4, 6, 7};
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - target; i++) {
            curSet.add(nums[i]);
            fun(nums, target, i + 1);
            curSet.remove(nums[i]);
        }

        System.out.println(ans % mode);
    }

    private static void fun(int[] nums, int target, int curIndex) {
        if (curIndex == nums.length) {
            if (curSet.size() >= target) {
                long temp = 1;
                int m = curSet.size();
                int n = target;
                for (int i = 1, j = m - n + 1; i <= n; i++, j++) {
                    temp = temp * j / i;
                }
                ans += (int) (temp % mode);
            }
            return;
        }
        boolean flag = true;
        for (int i = curIndex; i < nums.length; i++) {
            flag = true;
            for (Integer curNum : curSet) {
                if (nums[i] % curNum != 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                curSet.add(nums[i]);
                fun(nums, target, i + 1);
                curSet.remove(nums[i]);
            }

        }


    }

}

