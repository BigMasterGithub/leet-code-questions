package written.examination.huawei_9_27;

import java.util.LinkedList;
import java.util.Scanner;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/27 19:05
 **/
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int[] nums = new int[len];
        for (int i = 0; i <len ; i++) {
            nums[i] = in.nextInt();
        }



        int ans = fun_huawei(nums);
//        int ans = fun_huawei(new int[]{9,4,5,2,4});
        System.out.println(ans);
    }

    private static int fun_huawei(int[] nums) {
        int ans = 0 ;
        LinkedList<Integer> stack = new LinkedList<>();
        stack.add(nums[0]);
        for (int i = 1; i < nums.length; i++) {
            for(int j = i-1 ; j >=0 ; j--){
                if(nums[j] <= nums[i]){
                    ans+=nums[j];
                    break;
                }
            }
        }
        return ans;
    }
}

