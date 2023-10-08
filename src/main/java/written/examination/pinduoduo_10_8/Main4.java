package written.examination.pinduoduo_10_8;

import java.util.HashMap;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/10/8 16:10
 **/
public class Main4 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int i = in.nextInt();
        while (i-- > 0) {
            int n = in.nextInt();
            int t = in.nextInt();
            int[] times = new int[n];
            for (int j = 0; j < n; j++) {
                times[j] = in.nextInt();
            }
            int ans = fun(n, t, times);
            System.out.println(ans);
        }


    }

    public static int fun(int n, int t, int[] times) {
        int min_gap = 0;
        HashMap<Integer, Integer> map = new HashMap<>();

        int i = 0;
        int L = 0;
        int R = 0;
        while (i < n) {

            if (times[i] > t) {
                L = i;
                R = i + 1;
                while (R < n && times[R] > t) {
                    R++;
                }
                map.put(L, R - L);
                min_gap = Math.max(R - L, min_gap);
                i = R;
            } else {
                i++;
            }


        }


        return min_gap;
    }
}

