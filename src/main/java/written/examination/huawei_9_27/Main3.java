package written.examination.huawei_9_27;

import java.util.Arrays;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/27 20:03
 **/
public class Main3 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len_weights=in.nextInt();
        int len_loads = in .nextInt();
        int number = in.nextInt();
        int weighOfLoudou = in.nextInt();
        int[] weights = new int[len_weights];
        for (int i = 0; i < len_weights; i++) {
            weights[i]=in.nextInt();
        }
        int[] loads = new int[len_loads];

        for (int i = 0; i < len_loads; i++) {
            loads[i]=in.nextInt();
        }

        int ans = fun_3(weights, loads, number, weighOfLoudou);
        System.out.println(ans);
    }

    private static int fun_3(int[] weights, int[] loads, int number, int weighOfLoudou) {
        Arrays.sort(weights);
        Arrays.sort(loads);
        boolean[] isUsed = new boolean[loads.length];
        int ans = 0;

        for (int i = 0; i < weights.length; i++) {
            boolean isNeedLoudo = true;
            for (int j = 0; j < loads.length; j++) {

                if (loads[j] >= weights[i] && !isUsed[j]) {
                    ans++;
                    isUsed[j] = true;
                    isNeedLoudo = false;
                    break;
                }
            }
            if (isNeedLoudo) {
                for (int j = 0; j < loads.length; j++) {
                    if (number > 0 && loads[j] + weighOfLoudou >= weights[i]) {
                        ans++;
                        number--;
                        isUsed[j] = true;
                        break;
                    }
                }
            }
        }
        return ans;

    }
}

