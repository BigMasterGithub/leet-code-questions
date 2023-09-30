package written.examination.jingdong_9_16;

import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/16 10:42
 **/
public class Main3 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        int x = in.nextInt();
        int target = in.nextInt();
        fun(x,target);

    }

    private static void fun(int x, int target) {
        String x_binary = Integer.toBinaryString(x);
    }
}

