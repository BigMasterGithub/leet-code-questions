package written.examination.pinduoduo_10_8;

import java.util.Arrays;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/10/8 15:10
 **/
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int[] myself = new int[len];
        int[] duishou = new int[len];
        for (int i = 0; i < len; i++) {
            myself[i] = in.nextInt();
        }
        for (int i = 0; i < len; i++) {
            duishou[i] = in.nextInt();
        }
        int ans = fun(myself, duishou, len);
        System.out.println(ans);

    }

    private static int fun(int[] myself, int[] duishou, int len) {
        Arrays.sort(myself);
        // Arrays.sort(duishou);
        int ans = 0;
        // if (myself[len - 1] <= duishou[0]) return 0;

        boolean[] used = new boolean[len];
        Arrays.fill(used, false);
        int curMinIndex = 0;
        for (int i = 0; i < len; i++) {
            int j;
            for (j = curMinIndex; j < len; j++) {
                if (myself[j] > duishou[i] && !used[j]) {
                    ans++;
                    used[j] = true;
                    break;
                }
            }
            if (j == len) {
                while (curMinIndex < len && used[curMinIndex]) {
                    curMinIndex++;
                }
                if (curMinIndex < len)
                    used[curMinIndex] = true;

            }
        }
        return ans;
    }
}

