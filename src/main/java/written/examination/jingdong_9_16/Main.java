package written.examination.jingdong_9_16;

import java.util.Arrays;
import java.util.Scanner;

// 注意类名必须为 Main, 不要有任何 package xxx 信息
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);

        int len = in.nextInt();
        int floorLen = in.nextInt();
        int H = in.nextInt();
        int[] heights = new int[len];
        for (int i = 0; i < len; i++) {
            heights[i] = in.nextInt();
        }
//        System.out.println(Arrays.toString(heights));
        int[] add = new int[len];
        for (int i = 0; i < len; i++) {
            add[i] = in.nextInt() - 1;
        }
        //System.out.println(Arrays.toString(add));
        int[] floors = new int[floorLen];
        for (int i = 0; i < floorLen; i++) {
            floors[i] = in.nextInt();
        }
        for (int i = 0; i < len; i++) {
            heights[i] += floors[add[i]];
        }
        //System.out.println(Arrays.toString(heights));
        //System.out.println(Arrays.toString(floors));
        int ans = fun(heights, floors, H);
        System.out.println(ans);
    }

    private static int fun(int[] heights, int[] floors, int H) {
        int[] temp = Arrays.copyOf(floors, floors.length);
        Arrays.sort(temp);
        int maxHeight = temp[temp.length - 1];
        int curMaxHeight = H + maxHeight;
        int ans = 0;
        for (int i = 0; i < heights.length; i++) {
            if (curMaxHeight > heights[i]) ans++;
        }
        return ans;
    }

}
