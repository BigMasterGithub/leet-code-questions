package written.examination.zijie_10_6;

import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/10/8 19:05
 **/
public class Main {


    public static void main(String[] args) {
        System.out.println(fun(new int[]{3,3,3,1,2,1,1,2,3,3,4}));
    }

    static int fun(int[] fruits) {
        Set<Integer> set = new HashSet<>();
        int ans = 2;
        for (int step = fruits.length; step >= 2; step--) {
//            System.out.println("当前step ="+step);
            for (int i = 0; i + step-1 < fruits.length; i++) {
//                System.out.println("当前 i="+i);
                set.clear();
                for (int j = i; j  < i+step; j++) {
//                    System.out.println("当前 j="+j);
                    if (set.contains(fruits[j])){

                        if (j == i + step - 1) {
                            ans = step;
                            return ans;
                        }else{
                            continue;
                        }
                    }
                    else {
                        set.add(fruits[j]);
                        System.out.println(set.size());
                        if (set.size() > 2) break;
                    }

                }


            }
        }
        return ans;


    }

}


