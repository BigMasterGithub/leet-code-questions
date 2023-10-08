package written.examination.pinduoduo_10_8;

import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/10/8 15:26
 **/
public class Main3 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int i = in.nextInt();
        while (i-- > 0) {
            int m = in.nextInt();
            in.nextLine();
            String line = in.nextLine();
            int ans = fun(m, line, "PDD");
            System.out.println(ans);
        }
    }

    private static int fun(int m, String line, String target) {
        if(m>=3) return 2;

        return 5;
    }
}

