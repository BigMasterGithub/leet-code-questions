package written.examination.meituan_9_3;

import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/2 19:46
 **/
public class Main5 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int[] colors = new int[len];
        for (int i = 0; i < len; i++) {
            colors[i] = in.nextInt();
        }
        int ans = fun(colors);
        System.out.println(ans);
    }

    private static int fun(int[] colors) {

        return 0;
    }

}

