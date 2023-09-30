package written.examination.meituan_9_3;

import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/2 19:03
 **/
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

       int len = in.nextInt();
        if (len <= 0) return;
        int[] arr = new int[len];
        for (int i = 0; i < len; i++) {
            arr[i] = in.nextInt();
        }
//        int[] arr = new int[]{1, 3, 3};
        boolean ans = fun(arr);
        if (ans) {
            System.out.println("Yes");
        } else {
            System.out.println("No");
        }
    }


    private static boolean fun(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] <= arr[i - 1]) return false;
        }
        int[] arr2 = new int[arr.length - 1];
        for (int i = 1; i < arr.length; i++) {
            arr2[i - 1] = arr[i] - arr[i - 1];
        }
//        System.out.println(Arrays.toString(arr2));
        for (int i = 1; i < arr2.length; i++) {
            if (arr2[i] >= arr2[i - 1]) return false;
        }
        return true;
    }
}

