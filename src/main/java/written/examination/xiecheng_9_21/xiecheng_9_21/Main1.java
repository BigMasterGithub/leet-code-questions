package written.examination.xiecheng_9_21.xiecheng_9_21;

import java.util.*;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/9/21 19:06
 **/
public class Main1 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int len = in.nextInt();
        int[] nums = new int[len];
        for (int i = 0; i < len; i++) {
            nums[i] = in.nextInt();
        }
        System.out.println(Arrays.toString(nums));
        boolean[] visited = new boolean[len];
        Arrays.fill(visited, false);
        int[] ans = new int[len];
        PriorityQueue<Integer> queue = new PriorityQueue<>();
        for (int i = 1; i <= len; i++) {
            queue.add(i);
        }

        for (int i = 0; i < len; i++) {
            int curMinValue = queue.peek();
            if (curMinValue < nums[i]) {
                ans[i] = curMinValue;
                queue.poll();
            }else if(curMinValue == nums[i]){
                ans[i] = nums[i]+1;
            }

        }

    }
}

