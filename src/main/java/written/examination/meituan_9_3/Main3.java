package written.examination.meituan_9_3;

import java.util.Scanner;
import java.util.TreeSet;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/2 19:46
 **/
public class Main3 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
       int len = in.nextInt();
        int firstNum = in.nextInt();
        TreeSet<Integer> set = new TreeSet<>((Integer o1, Integer o2) -> {
            return o2 - o1;
        });
        for (int i = 1; i < len; i++) {
            set.add(in.nextInt());
        }
      /*  set.add(123);
        set.add(234);
        set.add(456);
        set.add(156);
        set.add(198);*/


//        int firstNum = 15;
        int ans = fun(set, firstNum);
        System.out.println(ans);

    }

    private static int fun(TreeSet<Integer> set, int firstNum) {
        Integer secondNum = 0;
        for (Integer cur : set) {
            secondNum = cur;
            break;
        }
//        System.out.println(secondNum);
        if (firstNum >= secondNum) return 0;
        int ans = 0;
        while(secondNum > firstNum){
            ans ++;
            secondNum/=2;
        }

        return ans;
    }

}

