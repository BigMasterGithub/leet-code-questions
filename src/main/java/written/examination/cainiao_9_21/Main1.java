package written.examination.cainiao_9_21;

import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/21 16:50
 **/
public class Main1 {
    public static void main(String[] args) {

        System.out.println(new Random().nextInt(999));

        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int attackInitial = in.nextInt();
//        int[][] dp = new int[len][2];
//        dp[0] = attackInitial;
//        dp[0][1] = 2 * attackInitial;
        int[] nums = new int[len];
        for (int i = 0; i < len; i++) {
            nums[i] = in.nextInt();
        }
        Arrays.sort(nums);
        System.out.println(Arrays.toString(nums));
        int rest = 1;
        for (int i = 0; i < len; i++) {
            if (attackInitial >= nums[i]) {
                attackInitial += nums[i] / 5;
                System.out.println("击杀该怪，增加攻击力" + nums[i] / 5 + ",当前攻击力为：" + attackInitial);
            } else if (attackInitial * 2 >= nums[i]) {

                if (rest >= 1) {
                    attackInitial += (attackInitial + nums[i] / 5);

                }
                rest--;
            } else {
                break;
            }
        }
        if (rest == 1) {

            System.out.println(attackInitial *= 2);
        } else {
            System.out.println(attackInitial);
        }
    }
}

