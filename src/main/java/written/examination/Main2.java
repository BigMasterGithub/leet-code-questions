package written.examination;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Main2 {
    static List<List<Integer>> ans;
    static List<Integer> temp;

    public static void main(String[] args) throws InterruptedException {
        Scanner in = new Scanner(System.in);
       int len = in.nextInt();
        int[] array = new int[len];
        for (int i = 0; i < len; i++) {
            array[i] = in.nextInt();
        }
        int target = in.nextInt();
        Arrays.sort(array);

        ans =new ArrayList<>();
        temp=new ArrayList<>();

        Arrays.sort(array);
        fun(array, target, 0, 0);
        System.out.println(ans);
    }

    //eg: 1,1,2,3,4,5  target = 5
    //eg: 1 1 2 3 4 5 5
    public static void fun(int[] array, int target, int index, int curSum) {
//        System.out.println("index = " + index);
        if (curSum == target) {
            ans.add(new ArrayList<>(temp));
            return;
        }
        if (index == array.length) {
            if (curSum == target) {
                ans.add(new ArrayList<>(temp));
            }
            return;
        }

        //不选择
        fun(array, target, index + 1, curSum);
        //选择当前元素
        if (curSum + array[index] <= target) {
            temp.add(array[index]);
            fun(array, target, index + 1, curSum + array[index]);
            temp.remove(temp.size() - 1);
        }

    }

}
