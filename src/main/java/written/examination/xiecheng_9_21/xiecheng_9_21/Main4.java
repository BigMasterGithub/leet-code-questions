package written.examination.xiecheng_9_21.xiecheng_9_21;

import java.util.ArrayList;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/21 19:53
 **/
public class Main4 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        String target = in.nextLine();
        int k = in.nextInt();
        int mode = 1_000_000_000 + 7;
        int ans = 0;
        Integer i = Integer.valueOf(target);
        if (target.length() <= 2) {

            System.out.println(0);
            return;
        }
        char[] charArray = target.toCharArray();
        if (charArray[charArray.length - 1] == '0' || charArray[charArray.length - 1] == '5') {
            fun(target.toCharArray(), k, 0, new ArrayList<>(), ans);
        } else {
            charArray[charArray.length - 1] = '0';
            fun(target.toCharArray(), k - 1, 0, new ArrayList<>(), ans);
            charArray[charArray.length - 1] = '5';
            fun(target.toCharArray(), k - 1, 0, new ArrayList<>(), ans);
        }
        System.out.println(ans%mode);


    }

    private static void fun(char[] charArray, int rest, int index, ArrayList<Object> temp, int ans) {
        if (index == charArray.length - 1) return;
        if (rest == 0) {
            ans++;
            return;
        }
        for (int i = 0; i <= 9; i++) {
            int cur = charArray[index] - '0';
            if (cur != i) {
                charArray[index] = (char) (i + '0');
                int curInteger = sum(charArray);
                if (curInteger % 75 == 0)
                    ans++;
                fun(charArray, rest - 1, index + 1, null, ans);
                charArray[index] = (char) (cur + '0');
            }


        }
    }

    private static int sum(char[] chars) {
        int ans = 0;
        for (int i = 0; i < chars.length; i++) {
            ans += (chars[i] - '0') * (int) Math.pow(10, i);
        }
        return ans;
    }
}

